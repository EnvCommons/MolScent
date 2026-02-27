"""
Molscent Environment - Matching scent/smell to molecules (multiple choice).

Following ether0's property-cat-smell pattern:
- Multiple choice: 4 SMILES options (A/B/C/D)
- Both positive ("which IS {scent}?") and negative ("which is NOT {scent}?") questions
- Same-family distractors from co-occurring scent descriptors
- Verification: exact string match on selected option

Binary reward: 1.0 if correct, else 0.0.
"""
import os
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, Field

from openreward.environments import Environment, JSONObject, Server, TextBlock, ToolOutput, tool


# ---- Data loading ----
if os.path.exists("/orwd_data"):
    DATA_PATH = Path("/orwd_data")
else:
    DATA_PATH = Path(__file__).parent

tasks_df = pd.read_parquet(DATA_PATH / "tasks.parquet")
if "__index_level_0__" in tasks_df.columns:
    tasks_df = tasks_df.drop("__index_level_0__", axis=1)
tasks_df = tasks_df.fillna("")
all_tasks = tasks_df.to_dict(orient="records")

for t in all_tasks:
    t["id"] = str(t["id"])

train_tasks = [t for t in all_tasks if t["split"] == "train"]
test_tasks = [t for t in all_tasks if t["split"] == "test"]


# ---- Pydantic models ----
class MolscentTaskSpec(BaseModel):
    id: str
    question_type: str
    target_scent: str
    correct_answer: str
    option_a: str
    option_b: str
    option_c: str
    option_d: str
    prompt: str
    split: str


class SubmitAnswerInput(BaseModel):
    answer: str = Field(..., description="Your answer: A, B, C, or D")


# ---- Environment class ----
class Molscent(Environment):
    """
    Molscent Environment: Multiple-choice scent-to-molecule matching.
    """

    def __init__(self, task_spec: JSONObject = {}, secrets: dict[str, str] = {}):
        super().__init__(task_spec)
        self.config = MolscentTaskSpec.model_validate(task_spec)

    @classmethod
    def list_splits(cls) -> list[str]:
        return ["train", "test"]

    @classmethod
    def list_tasks(cls, split: str) -> list[JSONObject]:
        if split == "train":
            return train_tasks
        elif split == "test":
            return test_tasks
        raise ValueError(f"Unknown split: {split}")

    def get_prompt(self) -> list[TextBlock]:
        return [TextBlock(type="text", text=self.config.prompt)]

    @tool
    async def submit_answer(self, params: SubmitAnswerInput) -> ToolOutput:
        """
        Submit your answer (A, B, C, or D) for the multiple-choice question.
        This finishes the episode.
        """
        submitted = params.answer.strip().upper()
        expected = self.config.correct_answer.strip().upper()

        # Normalize: accept "A)", "A.", "a" etc.
        if submitted and submitted[0] in "ABCD":
            submitted = submitted[0]

        correct = submitted == expected
        reward = 1.0 if correct else 0.0

        if correct:
            msg = f"Correct! The answer is {expected}."
        else:
            msg = f"Incorrect. You answered {submitted}, but the correct answer is {expected}."

        return ToolOutput(
            blocks=[TextBlock(type="text", text=msg)],
            metadata={
                "task_id": self.config.id,
                "question_type": self.config.question_type,
                "target_scent": self.config.target_scent,
                "submitted_answer": submitted,
                "correct_answer": expected,
                "correct": correct,
            },
            reward=reward,
            finished=True,
        )


if __name__ == "__main__":
    server = Server([Molscent])
    server.run()
