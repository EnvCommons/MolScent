# Molscent

OpenReward environment for matching scent/smell to molecules. Follows ether0's `property-cat-smell` pattern with multiple-choice questions and string-match verification.

## Task Format

Multiple-choice (A/B/C/D) with 4 SMILES options per question. Two question types:

### Positive (550 tasks)
"Which of the following molecules is likely a {scent} molecule?" — agent must pick the molecule that has the target scent descriptor.

### Negative (550 tasks)
"Which of the following molecules is likely to NOT be {scent}?" — agent must pick the molecule that does NOT have the target scent descriptor.

## Difficulty

Distractors are chosen from **same-family scent descriptors** (by co-occurrence in the GoodScents dataset), not random molecules. For example, a "vanilla" question uses distractors from "sweet", "creamy", "powdery" — molecules that share overlapping chemical space with vanilla-scented compounds but are not themselves labeled as vanilla.

## Dataset

- **Source**: Pyrfume GoodScents (~4,272 validated molecules, 251 scent descriptors with 10+ molecules each)
- **Split**: 1000 train / 100 test
- **Verification**: Exact string match on selected option (A/B/C/D)

## Local Development

```bash
# Generate dataset
pip install requests pandas pyarrow rdkit
python generate_dataset.py

# Run server
pip install -r requirements.txt
python server.py

# Test
python test_agent.py
```

## Docker

```bash
docker build -t molscent:test .
docker run -v $(pwd):/orwd_data -p 8080:8080 molscent:test
```

## Deployment

Namespace: `EnvCommons/molscent`

See [DATA_UPLOAD.md](DATA_UPLOAD.md) for cloud storage upload instructions.
