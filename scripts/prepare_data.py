"""
Data preparation pipeline for Askllama-reddit.

Reads raw custjsonl.jsonl, deduplicates entries, creates a proper
conversational prompt format using all three fields (title, post_content,
comments), and writes train/val splits to data/.
"""

from __future__ import annotations

import json
import random
import os
from pathlib import Path

RAW_DATA_PATH = Path(__file__).resolve().parent.parent / "custjsonl.jsonl"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data"
TRAIN_PATH = OUTPUT_DIR / "train.jsonl"
VAL_PATH = OUTPUT_DIR / "val.jsonl"

VAL_RATIO = 0.1
MIN_COMMENT_LENGTH = 10
RANDOM_SEED = 42


def format_prompt(title: str, post_content: str, comments: str) -> str:
    """Create a conversational prompt combining all three data fields."""
    prompt = (
        f"### Post Title:\n{title.strip()}\n\n"
        f"### Post Content:\n{post_content.strip()}\n\n"
        f"### Top Comments:\n{comments.strip()}"
    )
    return prompt


def load_raw_data(path: Path) -> list[dict]:
    """Load JSONL file, return list of dicts."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"  Skipping malformed JSON at line {line_num}")
    return records


def deduplicate(records: list[dict]) -> list[dict]:
    """Remove exact duplicates based on (title, post_content, comments)."""
    seen = set()
    unique = []
    for r in records:
        key = (r.get("title", ""), r.get("post_content", ""), r.get("comments", ""))
        if key not in seen:
            seen.add(key)
            unique.append(r)
    return unique


def filter_records(records: list[dict]) -> list[dict]:
    """Filter out entries with empty or very short comments."""
    filtered = []
    for r in records:
        comments = r.get("comments", "").strip()
        if len(comments) >= MIN_COMMENT_LENGTH:
            filtered.append(r)
    return filtered


def add_formatted_text(records: list[dict]) -> list[dict]:
    """Add a 'text' field with the full formatted prompt."""
    for r in records:
        r["text"] = format_prompt(
            r.get("title", ""),
            r.get("post_content", ""),
            r.get("comments", ""),
        )
    return records


def split_train_val(records: list[dict], val_ratio: float, seed: int):
    """Randomly split records into train and validation sets."""
    random.seed(seed)
    shuffled = records.copy()
    random.shuffle(shuffled)
    val_size = max(1, int(len(shuffled) * val_ratio))
    return shuffled[val_size:], shuffled[:val_size]


def write_jsonl(records: list[dict], path: Path):
    """Write records to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    print("=== Askllama-reddit Data Preparation ===\n")

    # Load
    print(f"Loading raw data from {RAW_DATA_PATH}...")
    raw = load_raw_data(RAW_DATA_PATH)
    print(f"  Raw records: {len(raw)}")

    # Deduplicate
    unique = deduplicate(raw)
    print(f"  After deduplication: {len(unique)}")

    # Filter
    filtered = filter_records(unique)
    print(f"  After filtering (comments >= {MIN_COMMENT_LENGTH} chars): {len(filtered)}")

    # Format
    formatted = add_formatted_text(filtered)

    # Split
    train, val = split_train_val(formatted, VAL_RATIO, RANDOM_SEED)
    print(f"\n  Train set: {len(train)}")
    print(f"  Validation set: {len(val)}")

    # Write
    write_jsonl(train, TRAIN_PATH)
    write_jsonl(val, VAL_PATH)
    print(f"\n  Written to:")
    print(f"    {TRAIN_PATH}")
    print(f"    {VAL_PATH}")

    # Preview
    print(f"\n=== Sample formatted entry ===\n")
    print(formatted[0]["text"][:500])
    print("..." if len(formatted[0]["text"]) > 500 else "")

    print("\nDone!")


if __name__ == "__main__":
    main()
