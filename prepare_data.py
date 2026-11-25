"""
Step 1: Build the dataset (JSONL + CSV) with input, output and metadata fields,
then create a stratified 70/15/15 train/validation/test split.

Usage:  python prepare_data.py
"""
import json
import re
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

from config import DATA_DIR, SEED, TEST_FRAC, VAL_FRAC

sys.path.insert(0, str(DATA_DIR))
from raw_samples import LOCAL_TERMS, SAMPLES  # noqa: E402
from raw_samples_extra import EXTRA  # noqa: E402

SAMPLES = {k: SAMPLES[k] + EXTRA[k] for k in SAMPLES}  # batch 1 + batch 2

PHONE = re.compile(r"(\+?254|0)7\d{8}")
EMAIL = re.compile(r"\S+@\S+")


def anonymize(text: str) -> str:
    """Mask phone numbers and e-mails (none expected, but enforced for safety)."""
    text = PHONE.sub("[PHONE]", text)
    return EMAIL.sub("[EMAIL]", text)


def has_local_term(text: str) -> bool:
    t = text.lower()
    return any(term in t for term in LOCAL_TERMS)


def main():
    rows = []
    i = 1
    for label, items in SAMPLES.items():
        for text, lang in items:
            text = anonymize(text.strip())
            rows.append({
                "id": f"FQ{i:04d}",
                "input": text,
                "output": label,
                "language": lang,
                "has_local_term": has_local_term(text),
                "source": "ai_assisted_draft_reviewed_by_student",
                "n_words": len(text.split()),
            })
            i += 1
    df = pd.DataFrame(rows).drop_duplicates(subset="input")

    # Stratify on label so every split has all 6 classes in equal proportion
    train, temp = train_test_split(df, test_size=VAL_FRAC + TEST_FRAC,
                                   stratify=df["output"], random_state=SEED)
    val, test = train_test_split(temp, test_size=TEST_FRAC / (VAL_FRAC + TEST_FRAC),
                                 stratify=temp["output"], random_state=SEED)
    df["split"] = "train"
    df.loc[val.index, "split"] = "val"
    df.loc[test.index, "split"] = "test"

    # Full dataset in the required input / output / metadata format
    with open(DATA_DIR / "farmer_queries.jsonl", "w", encoding="utf-8") as f:
        for _, r in df.iterrows():
            rec = {
                "input": r["input"],
                "output": r["output"],
                "metadata": {k: (bool(r[k]) if k == "has_local_term" else r[k])
                             for k in ["id", "language", "has_local_term", "source", "split", "n_words"]},
            }
            f.write(json.dumps(rec, ensure_ascii=False, default=int) + "\n")
    df.to_csv(DATA_DIR / "farmer_queries.csv", index=False)
    for name in ["train", "val", "test"]:
        df[df.split == name].to_csv(DATA_DIR / f"{name}.csv", index=False)

    print(f"Total samples: {len(df)}")
    print(pd.crosstab(df["output"], df["split"], margins=True))
    print(pd.crosstab(df["language"], df["split"], margins=True))


if __name__ == "__main__":
    main()
