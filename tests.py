import json
import re

import pytest

from server import Molscent, SubmitAnswerInput, _extract_letter

TASKS = Molscent.list_tasks("train")


def _wrong_letter(task) -> str:
    return next(x for x in "ABCD" if x != task["correct_answer"])


@pytest.mark.asyncio
@pytest.mark.parametrize("task", TASKS[:5], ids=lambda t: t["id"])
async def test_gold(task):
    env = Molscent(task_spec=task)
    result = await env.submit_answer(SubmitAnswerInput(answer=f"Answer: {task['correct_answer']}"))
    assert result.reward == 1.0
    assert result.finished


@pytest.mark.asyncio
@pytest.mark.parametrize("task", TASKS[:5], ids=lambda t: t["id"])
async def test_xfail(task):
    env = Molscent(task_spec=task)
    result = await env.submit_answer(SubmitAnswerInput(answer=f"Answer: {_wrong_letter(task)}"))
    assert result.reward == 0.0
    assert result.finished


@pytest.mark.asyncio
@pytest.mark.parametrize("task", TASKS[:20], ids=lambda t: t["id"])
async def test_incorrect_feedback_does_not_reveal_answer(task):
    env = Molscent(task_spec=task)
    result = await env.submit_answer(SubmitAnswerInput(answer=_wrong_letter(task)))
    assert result.reward == 0.0
    text = " ".join(b.text for b in result.blocks)
    assert task["correct_answer"] not in re.findall(r"(?<![A-Za-z])[ABCD](?![A-Za-z])", text)
    assert "correct_answer" not in result.metadata
    assert task["correct_answer"] not in result.metadata.values()


@pytest.mark.asyncio
async def test_no_letter_is_graded_zero_with_feedback():
    env = Molscent(task_spec=TASKS[0])
    result = await env.submit_answer(SubmitAnswerInput(answer="I am not sure which one it is."))
    assert result.reward == 0.0
    assert result.finished
    assert "did not state an option letter" in result.blocks[0].text


@pytest.mark.asyncio
async def test_repeat_submission_is_penalised():
    task = TASKS[0]
    env = Molscent(task_spec=task)
    await env.submit_answer(SubmitAnswerInput(answer=task["correct_answer"]))
    result = await env.submit_answer(SubmitAnswerInput(answer=task["correct_answer"]))
    assert result.reward < 0
    assert result.finished


ANSWER_HEADER_THEN_LIST = """Both reviewers agree with the first reading.

**Answer: C**

- **A) `CCC(C)(CCc1ccccc1)OC(C)=O`** - an aromatic acetate ester.
- **B) `CC(C)C(=O)OCCCc1ccccc1`** - another aromatic ester.
- **C) `CCCCCCCCC(CC)C(=O)O`** - a long-chain branched carboxylic acid.
- **D) `CCCCCCCCCCCCOC(=O)CCSCCC(=O)OCCCCCCCCCCCC`** - a sulfur-containing di-ester.

**C** is the only option that is a fatty acid, which is a hallmark of soap."""

LEADING_LABEL_THEN_CONTRAST = """A) CC(=O)OC(C)C(C)=O

This structure carries a ketone (the terminal `C(C)=O`) next to an acetate ester. The others lack one: B has no carbonyl, C has an aldehyde, and D only has ester carbonyls."""

BOLD_CHOICE_THEN_PROSE = """The most probable candidate is **C**: `CCCCCCCCC(CC)C(=O)O`, a branched fatty acid. The other options are aromatic esters or a large sulfur-containing di-ester with a probable off-note."""

OPTION_FIRST_THEN_CONTRAST = """Option C is the most probable match. `CCCCCCCCC(CC)C(=O)O` is a branched fatty acid, whereas the other options are aromatic esters (A and B) or a complex di-ester (D)."""

LABELLED_SMILES_IN_BOLD = """The molecule is **D) CC/C=C\\CCOC(C)OCC**.

It is an acetal carrying a green-leaf hexenyl group. A is an acetate ester, B is methyl salicylate, and C is an unsaturated ketone."""

FINAL_ANSWER_WITH_CAVEAT = """Only **B** has an aldehyde group, so any of the others qualifies; **A** is the cleanest case.

**Final answer: A** (C and D also lack an aldehyde; B is the aldehyde)."""


@pytest.mark.parametrize(
    "text, expected",
    [
        (ANSWER_HEADER_THEN_LIST, "C"),
        (LEADING_LABEL_THEN_CONTRAST, "A"),
        (BOLD_CHOICE_THEN_PROSE, "C"),
        (OPTION_FIRST_THEN_CONTRAST, "C"),
        (LABELLED_SMILES_IN_BOLD, "D"),
        (FINAL_ANSWER_WITH_CAVEAT, "A"),
        ("The answer is (B).", "B"),
        ("After weighing the options, $\\boxed{D}$.", "D"),
        ("D", "D"),
        ("**B**", "B"),
        ("C. It is the only ester.", "C"),
    ],
    ids=[
        "answer-header-then-list",
        "leading-label-then-contrast",
        "bold-choice-then-prose",
        "option-first-then-contrast",
        "labelled-smiles-in-bold",
        "final-answer-with-caveat",
        "answer-is-parenthesised",
        "boxed",
        "bare-letter",
        "bare-bold-letter",
        "letter-then-sentence",
    ],
)
def test_extract_letter(text, expected):
    assert _extract_letter(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "It is a long-chain acid with a carboxyl group.",
        "The SMILES `OC(C)C(=O)OC` is an ester.",
        "",
    ],
    ids=["lowercase-article", "smiles-only", "empty"],
)
def test_extract_letter_none(text):
    assert _extract_letter(text) == ""


@pytest.mark.asyncio
async def test_extraction_feeds_grading():
    task = next(t for t in TASKS if t["correct_answer"] == "C")
    env = Molscent(task_spec=task)
    result = await env.submit_answer(SubmitAnswerInput(answer=ANSWER_HEADER_THEN_LIST))
    assert result.reward == 1.0
    assert result.metadata["submitted_answer"] == "C"
    json.dumps(result.metadata)
