# Data Upload Requirements for molscent

## Overview

The molscent environment requires one data file uploaded to OpenReward cloud storage.

## Directory Structure

```
/orwd_data/
└── tasks.parquet (~150 KB, 1100 rows)
```

## File Description

- **tasks.parquet**: Task definitions with columns:
  - `id` (str): Task identifier, e.g. "ms_0000"
  - `question_type` (str): "positive" or "negative"
  - `target_scent` (str): Target scent descriptor (e.g. "vanilla", "woody")
  - `correct_answer` (str): Correct option letter (A, B, C, or D)
  - `option_a` through `option_d` (str): SMILES strings for each option
  - `prompt` (str): Pre-rendered prompt text with options
  - `split` (str): "train" (1000) or "test" (100)

## Data Source

Tasks generated from Pyrfume GoodScents dataset:
- https://github.com/pyrfume/pyrfume-data/tree/main/goodscents
- ~4,272 validated molecules with 251 scent descriptors (filtered to 10+ molecules each)
- Distractors selected from co-occurring scent families for difficulty

## Regenerating

```bash
pip install requests pandas pyarrow rdkit
python generate_dataset.py
```

## Upload Instructions

Upload `tasks.parquet` to the root of the OpenReward namespace storage
for EnvCommons/molscent at https://openreward.ai.
