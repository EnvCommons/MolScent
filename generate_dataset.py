"""
generate_dataset.py - Generate molscent multiple-choice tasks from Pyrfume GoodScents data.

Downloads behavior/stimuli/molecules CSVs from pyrfume-data GitHub,
joins them, and produces tasks.parquet.

Task format (following ether0's property-cat-smell pattern):
- Multiple choice: 4 SMILES options (A/B/C/D)
- One correct answer (molecule labeled with target scent)
- Three same-family distractors (molecules from co-occurring scents, NOT labeled with target scent)
- Both positive ("which IS {scent}?") and negative ("which is NOT {scent}?") questions

Dependencies: requests, pandas, pyarrow, rdkit
"""
import json
import random
from collections import Counter, defaultdict
from io import StringIO
from pathlib import Path

import pandas as pd
import requests
from rdkit import Chem

OUTPUT_DIR = Path(__file__).parent
TASKS_PATH = OUTPUT_DIR / "tasks.parquet"

BASE_URL = "https://raw.githubusercontent.com/pyrfume/pyrfume-data/main/goodscents"

TARGET_TOTAL = 1100
MIN_MOLECULES_PER_SCENT = 10
MAX_DESCRIPTOR_WORDS = 3
NUM_OPTIONS = 4
OPTION_LABELS = ["A", "B", "C", "D"]

# ether0-style prompt templates with {rel} for positive/negative
PROMPT_TEMPLATES = [
    "Which of the following options likely is{rel} a {property} molecule?\n{options}",
    "Which of the following molecules is likely to{rel} be {property}?\n{options}",
    "Identify the molecule below that likely is{rel} a {property} molecule:\n{options}",
    "From the list below, select the molecule most likely to{rel} be {property}:\n{options}",
    "Choose the molecule from the options below that most probably is{rel} {property}:\n{options}",
    "Among the following, which molecule likely is{rel} considered {property}?\n{options}",
    "Select the molecule below most expected to{rel} have {property} properties:\n{options}",
    "From these molecules, identify the one most likely to{rel} possess {property}:\n{options}",
    "Which candidate below most probably is{rel} classified as a {property} molecule?\n{options}",
]


def download_csv(filename: str) -> pd.DataFrame:
    url = f"{BASE_URL}/{filename}"
    print(f"  Downloading {url}...")
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return pd.read_csv(StringIO(resp.text))


def format_options(option_smiles: list[str]) -> str:
    """Format SMILES options as A) ... B) ... etc."""
    lines = []
    for label, smi in zip(OPTION_LABELS, option_smiles):
        lines.append(f"{label}) {smi}")
    return "\n".join(lines)


def main():
    random.seed(42)

    # ---- Step 1: Download CSVs ----
    print("Downloading Pyrfume GoodScents data...")
    behavior_df = download_csv("behavior.csv")
    stimuli_df = download_csv("stimuli.csv")
    molecules_df = download_csv("molecules.csv")

    print(f"  behavior: {len(behavior_df)} rows")
    print(f"  stimuli: {len(stimuli_df)} rows")
    print(f"  molecules: {len(molecules_df)} rows")

    # ---- Step 2: Join tables ----
    merged = behavior_df.merge(
        stimuli_df[["Stimulus", "CID"]], on="Stimulus", how="inner"
    )
    merged = merged.merge(
        molecules_df[["CID", "IsomericSMILES", "name"]], on="CID", how="inner"
    )
    print(f"  Merged: {len(merged)} rows")

    # ---- Step 3: Parse descriptors, validate SMILES ----
    records = []
    seen_canonical = set()

    for _, row in merged.iterrows():
        smiles = str(row.get("IsomericSMILES", "")).strip()
        if not smiles:
            continue

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            continue

        canonical = Chem.MolToSmiles(mol, canonical=True)
        if "." in canonical:
            continue

        num_heavy = mol.GetNumHeavyAtoms()
        if num_heavy < 3 or num_heavy > 100:
            continue

        if canonical in seen_canonical:
            continue
        seen_canonical.add(canonical)

        raw_desc = str(row.get("Descriptors", ""))
        descriptors = [d.strip().lower() for d in raw_desc.split(";") if d.strip()]
        descriptors = [d for d in descriptors if d and d != "odorless"]
        descriptors = [d for d in descriptors if len(d.split()) <= MAX_DESCRIPTOR_WORDS]

        if not descriptors:
            continue

        name = str(row.get("name", "")).strip()
        if not name:
            name = canonical

        records.append({
            "canonical": canonical,
            "name": name,
            "descriptors": set(descriptors),
        })

    print(f"  Valid molecules with scent descriptors: {len(records)}")

    # ---- Step 4: Build scent -> molecules index ----
    scent_to_mols: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        for desc in rec["descriptors"]:
            scent_to_mols[desc].append(rec)

    # Filter to descriptors with enough molecules
    filtered_scents = {
        s: mols for s, mols in scent_to_mols.items()
        if len(mols) >= MIN_MOLECULES_PER_SCENT
    }
    scent_names = sorted(filtered_scents.keys())
    print(f"  Descriptors with >= {MIN_MOLECULES_PER_SCENT} molecules: {len(filtered_scents)}")

    # ---- Step 5: Build co-occurrence map for same-family distractors ----
    print("Building co-occurrence map...")
    cooccur: dict[str, Counter] = defaultdict(Counter)
    for rec in records:
        descs = [d for d in rec["descriptors"] if d in filtered_scents]
        for d1 in descs:
            for d2 in descs:
                if d1 != d2:
                    cooccur[d1][d2] += 1

    # For each scent, get ranked list of related scents (by co-occurrence)
    related_scents: dict[str, list[str]] = {}
    for scent in scent_names:
        ranked = [s for s, _ in cooccur[scent].most_common() if s in filtered_scents]
        if not ranked:
            # Fallback: use all other scents
            ranked = [s for s in scent_names if s != scent]
        related_scents[scent] = ranked

    # ---- Step 6: Build canonical -> descriptors lookup ----
    mol_descriptors: dict[str, set[str]] = {}
    for rec in records:
        mol_descriptors[rec["canonical"]] = rec["descriptors"] & set(scent_names)

    # ---- Step 7: Generate tasks ----
    print("Generating multiple-choice tasks...")
    tasks = []
    task_idx = 0

    # Split roughly 50/50 between positive and negative questions
    n_positive = TARGET_TOTAL // 2
    n_negative = TARGET_TOTAL - n_positive

    # Distribute across scents
    tasks_per_scent = max(1, TARGET_TOTAL // len(scent_names))

    for question_type, target_count in [("positive", n_positive), ("negative", n_negative)]:
        remaining = target_count
        shuffled = scent_names.copy()
        random.shuffle(shuffled)

        scent_idx = 0
        attempts = 0
        max_attempts = target_count * 10

        while remaining > 0 and attempts < max_attempts:
            attempts += 1
            scent = shuffled[scent_idx % len(shuffled)]
            scent_idx += 1

            positives = filtered_scents[scent]
            if len(positives) < 1:
                continue

            # Pick the correct molecule
            correct_mol = random.choice(positives)

            # Pick 3 same-family distractors: molecules from co-occurring scents
            # that do NOT have the target scent label
            distractor_pool = []
            for related in related_scents[scent]:
                for mol_rec in filtered_scents[related]:
                    if scent not in mol_rec["descriptors"] and mol_rec["canonical"] != correct_mol["canonical"]:
                        distractor_pool.append(mol_rec)

            # Deduplicate by canonical
            seen = {correct_mol["canonical"]}
            unique_distractors = []
            for d in distractor_pool:
                if d["canonical"] not in seen:
                    seen.add(d["canonical"])
                    unique_distractors.append(d)

            if len(unique_distractors) < NUM_OPTIONS - 1:
                # Fallback: add random molecules without the target scent
                for rec in records:
                    if rec["canonical"] not in seen and scent not in rec["descriptors"]:
                        seen.add(rec["canonical"])
                        unique_distractors.append(rec)
                    if len(unique_distractors) >= NUM_OPTIONS - 1:
                        break

            if len(unique_distractors) < NUM_OPTIONS - 1:
                continue

            distractors = random.sample(unique_distractors, NUM_OPTIONS - 1)

            if question_type == "positive":
                # Positive: correct answer IS the target-scent molecule
                # Place correct answer at random position
                correct_pos = random.randint(0, NUM_OPTIONS - 1)
                options = list(distractors)
                options.insert(correct_pos, correct_mol)
                correct_answer = OPTION_LABELS[correct_pos]
                rel = ""
            else:
                # Negative: correct answer is NOT the target-scent molecule
                # The "correct" answer is one of the distractors (the odd one out)
                # We need: 3 molecules WITH the scent, 1 WITHOUT
                if len(positives) < NUM_OPTIONS - 1:
                    continue

                # Pick 3 positives and 1 distractor
                pos_sample = random.sample(positives, NUM_OPTIONS - 1)
                neg_mol = random.choice(distractors)

                correct_pos = random.randint(0, NUM_OPTIONS - 1)
                options = list(pos_sample)
                options.insert(correct_pos, neg_mol)
                correct_answer = OPTION_LABELS[correct_pos]
                rel = " not"

            option_smiles = [o["canonical"] for o in options]
            options_text = format_options(option_smiles)

            prompt = random.choice(PROMPT_TEMPLATES).format(
                rel=rel,
                property=scent,
                options=options_text,
            )

            prompt += "\n\nReply with your final answer as an ordinary message. State the letter A, B, C, or D."

            tasks.append({
                "id": f"ms_{task_idx:04d}",
                "question_type": question_type,
                "target_scent": scent,
                "correct_answer": correct_answer,
                "option_a": option_smiles[0],
                "option_b": option_smiles[1],
                "option_c": option_smiles[2],
                "option_d": option_smiles[3],
                "prompt": prompt,
            })
            task_idx += 1
            remaining -= 1

        print(f"  {question_type}: generated {target_count - remaining} tasks (attempts: {attempts})")

    # ---- Step 8: Shuffle and assign splits ----
    random.shuffle(tasks)

    for i, t in enumerate(tasks):
        t["id"] = f"ms_{i:04d}"
        t["split"] = "train" if i < 1000 else "test"

    df = pd.DataFrame(tasks)
    df.to_parquet(TASKS_PATH, index=False)

    print(f"\nSaved {len(df)} tasks to {TASKS_PATH}")
    print(f"  Train: {len(df[df['split'] == 'train'])}")
    print(f"  Test:  {len(df[df['split'] == 'test'])}")
    print(f"  Positive questions: {len(df[df['question_type'] == 'positive'])}")
    print(f"  Negative questions: {len(df[df['question_type'] == 'negative'])}")
    print(f"  Unique target scents: {df['target_scent'].nunique()}")

    # Print a few examples
    print("\n--- Sample tasks ---")
    for _, row in df.head(3).iterrows():
        print(f"\n  [{row['question_type']}] scent={row['target_scent']}, answer={row['correct_answer']}")
        print(f"  {row['prompt'][:200]}...")


if __name__ == "__main__":
    main()
