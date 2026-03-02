# MolScent

[![OpenReward Environment](https://img.shields.io/badge/%E2%AD%90%20OpenReward-Environment-f7e6cc)](https://openreward.ai/GeneralReasoning/MolScent)

## Description

**MolScent** is an environment for evaluating agents on matching scent descriptors to molecules. Given a multiple-choice question with four SMILES options, the agent must identify which molecule has (or does not have) a target scent property. The dataset is derived from the [Pyrfume GoodScents dataset](https://pyrfume.org/) (~4,272 validated molecules, 251 scent descriptors). Distractors are chosen from same-family scent descriptors by co-occurrence, making the task non-trivial.

## Capabilities

- Associating molecular structures (SMILES) with olfactory properties
- Multiple-choice reasoning over molecular scent descriptors
- Distinguishing molecules with similar chemical profiles but different scent properties

## Compute Requirements

MolScent does not require a sandbox. It has minimal compute requirements.

## License

[MIT](https://opensource.org/license/mit).

## Tasks

There are two splits: train (1,000 tasks) and test (100 tasks), totaling 1,100 tasks. Each task is a 4-option multiple-choice question (A/B/C/D) with two question types:

- **Positive** (~550 tasks): "Which molecule IS {scent}?" -- the agent picks the molecule with the target scent descriptor.
- **Negative** (~550 tasks): "Which molecule is NOT {scent}?" -- the agent picks the molecule that does NOT have the target scent descriptor.

## Reward Structure

This is a sparse, verifiable reward environment with binary scoring. The agent calls `submit_answer` once with a letter (A, B, C, or D). The answer is compared via exact match against the correct answer.

- **Correct**: Reward **1.0**.
- **Incorrect**: Reward **0.0**.

We do not use LLM graders for this task.

## Data

Tasks are generated from the [Pyrfume GoodScents dataset](https://pyrfume.org/) (~4,272 validated molecules with scent descriptors). Distractors are selected from co-occurring scent families to ensure difficulty. Data is stored as a parquet file on the OpenReward platform.

## Tools

Agents are given a single tool:

- `submit_answer`: Submit an answer letter (A, B, C, or D) for the multiple-choice question. Returns whether the answer is correct. This tool can only be called once per task.

## Time Horizon

MolScent is a single-turn environment. The agent receives a multiple-choice question and submits one answer. Each task requires exactly one tool call.

## Environment Difficulty

[Statistics on environment difficulty here]

## Other Environment Requirements

There are no further environment requirements; MolScent works out of the box with the OpenReward endpoint.

## Safety

Agents in MolScent are asked to match scent descriptors to molecular structures. The environment does not present direct safety risks, as agents only provide letter answers with no access to external systems.

## Citations

```bibtex
@dataset{GRMolScent,
  author    = {General Reasoning Inc. Team},
  title     = {MolScent},
  year      = {2026},
  publisher = {OpenReward},
  url       = {https://openreward.ai/GeneralReasoning/MolScent}
}
```
