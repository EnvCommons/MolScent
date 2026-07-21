"""test_agent.py - Local test agent for MolScent (terminal-tool style).

The environment uses a hidden @terminal tool: the agent's final plain-text
message ends the rollout and is graded by extracting the answer letter.

Uses OpenAI Responses API. Requires OPENAI_API_KEY.
"""
import asyncio
import json
import os
from datetime import datetime, timezone

from openai import AsyncOpenAI
from openreward import AsyncOpenReward

MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-5.2")
NUM_TASKS = int(os.environ.get("NUM_TASKS", "2"))
MAX_TURNS = int(os.environ.get("MAX_TURNS", "40"))
TRAJECTORY_PATH = "molscent_trajectory.jsonl"


def _text_of(response) -> str:
    parts = []
    for item in response.output:
        if item.type == "message":
            for block in item.content:
                if block.type == "output_text":
                    parts.append(block.text)
    return "\n".join(parts).strip()


async def main():
    or_client = AsyncOpenReward()
    oai_client = AsyncOpenAI()

    environment = or_client.environments.get(
        name="EnvCommons/molscent", base_url="http://localhost:8080"
    )
    tasks = await environment.list_tasks(split="train")
    tools = await environment.list_tools(format="openai")
    terminal_tool = await environment.terminal_tool()

    print(f"Number of tasks: {len(tasks)}")
    print(f"Visible tools: {[t['name'] for t in tools]}")
    print(f"Terminal tool: {terminal_tool}")

    traj = open(TRAJECTORY_PATH, "w")

    def record(kind: str, **fields):
        traj.write(json.dumps({
            "kind": kind,
            "ts": datetime.now(timezone.utc).isoformat(),
            **fields,
        }) + "\n")
        traj.flush()

    record("config", model=MODEL_NAME, visible_tools=[t["name"] for t in tools],
           terminal_tool=None if terminal_tool is None else {
               "name": terminal_tool.name, "arg": terminal_tool.arg,
           })

    rewards = []

    for task in tasks[:NUM_TASKS]:
        async with environment.session(task=task) as session:
            assistant_ends_rollout = await session.is_assistant_message_final()
            session_tools = await session.list_tools()
            assert "submit_answer" not in [t.name for t in session_tools], \
                "terminal tool leaked into the model's tool list"

            prompt = await session.get_prompt()
            input_list = [{"role": "user", "content": prompt[0].text}]
            print(f"\n=== Task {task.task_spec['id']} ===")
            print(f"Prompt: {prompt[0].text}\n")
            record("task_start", task_id=task.task_spec["id"],
                   expected_answer=task.task_spec["correct_answer"],
                   is_assistant_message_final=assistant_ends_rollout,
                   session_tools=[t.name for t in session_tools],
                   prompt=prompt[0].text)

            reward = None
            turn = 0
            while turn < MAX_TURNS:
                turn += 1
                response = await oai_client.responses.create(
                    model=MODEL_NAME, tools=tools, input=input_list,
                )
                input_list += response.output

                calls = [i for i in response.output if i.type == "function_call"]
                if calls:
                    for item in calls:
                        args = json.loads(str(item.arguments))
                        tool_result = await session.call_tool(item.name, args)
                        input_list.append({
                            "type": "function_call_output",
                            "call_id": item.call_id,
                            "output": tool_result.blocks[0].text,
                        })
                        record("tool_call", task_id=task.task_spec["id"], turn=turn,
                               tool=item.name, arguments=args,
                               output=tool_result.blocks[0].text,
                               reward=tool_result.reward, finished=tool_result.finished)
                    continue

                final_message = _text_of(response)
                print(f"Final message: {final_message[:200]}")
                record("assistant_final_message", task_id=task.task_spec["id"],
                       turn=turn, text=final_message)

                if not assistant_ends_rollout:
                    print("Environment is not terminal-tool style; stopping.")
                    break

                out = await session.call_terminal_tool(final_message)
                reward = out.reward
                print(f"call_terminal_tool -> reward={reward} finished={out.finished}")
                record("terminal_tool_result", task_id=task.task_spec["id"], turn=turn,
                       submitted=final_message, reward=out.reward,
                       finished=out.finished, output=out.blocks[0].text,
                       metadata=out.metadata)
                break

            rewards.append(reward)
            record("task_end", task_id=task.task_spec["id"], turns=turn, reward=reward)

    scored = [r for r in rewards if r is not None]
    summary = {
        "num_tasks": len(rewards),
        "num_scored": len(scored),
        "mean_reward": (sum(scored) / len(scored)) if scored else None,
        "rewards": rewards,
    }
    record("summary", **summary)
    traj.close()
    print(f"\n=== Summary ===\n{json.dumps(summary, indent=2)}")


if __name__ == "__main__":
    asyncio.run(main())
