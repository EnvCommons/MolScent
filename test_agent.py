"""
test_agent.py - Local test agent for molscent environment.

Uses OpenAI Responses API (modern format).
Requires OPENAI_API_KEY environment variable.
"""
import asyncio
import json
import os

from openai import OpenAI
from openreward import AsyncOpenReward

MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-5.2")


async def test_function():
    or_client = AsyncOpenReward()
    oai_client = OpenAI()

    environment = or_client.environments.get(
        name="EnvCommons/molscent", base_url="http://localhost:8080"
    )
    tasks = await environment.list_tasks(split="train")
    tools = await environment.list_tools(format="openai")

    print(f"Number of tasks: {len(tasks)}")
    print(f"Number of tools: {len(tools)}")

    example_task = tasks[0]

    async with environment.session(task=example_task) as session:
        prompt = await session.get_prompt()
        input_list = [{"role": "user", "content": prompt[0].text}]
        finished = False
        print(f"\nPrompt: {prompt[0].text}\n")

        while not finished:
            response = oai_client.responses.create(
                model=MODEL_NAME, tools=tools, input=input_list
            )
            print(f"Model output: {response.output}")
            input_list += response.output

            for item in response.output:
                if item.type == "function_call":
                    tool_result = await session.call_tool(
                        item.name, json.loads(str(item.arguments))
                    )
                    finished = tool_result.finished

                    input_list.append({
                        "type": "function_call_output",
                        "call_id": item.call_id,
                        "output": json.dumps({
                            "result": tool_result.blocks[0].text
                        }),
                    })

                    print(f"\nTool result: {tool_result.blocks[0].text}")
                    print(f"Reward: {tool_result.reward}")
                    print(f"Finished: {tool_result.finished}")

                    if tool_result.finished:
                        finished = True
                        break


if __name__ == "__main__":
    asyncio.run(test_function())
