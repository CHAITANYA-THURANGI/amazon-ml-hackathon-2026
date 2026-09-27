"""
Data inspection script — run from the project root.

Usage:
    python inspect_data.py
"""

import pandas as pd
from pathlib import Path
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent / "dataset"


def inspect_file(path):
    print("\n" + "=" * 80)
    print(path)
    print("=" * 80)

    df = pd.read_csv(path, sep="\t")

    print(f"Rows:    {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print(f"Memory:  {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

    print("\nColumns:")
    for col in df.columns:
        print(f"  {col:25} dtype={str(df[col].dtype):10} "
              f"unique={df[col].nunique(dropna=True):,} "
              f"missing={df[col].isna().sum():,}")

    print("\nFirst 3 rows:")
    print(df.head(3).to_string(index=False))

    if "country" in df.columns:
        print("\nCountry distribution:")
        print(df["country"].value_counts(dropna=False).head(20).to_string())

    if "business_name" in df.columns:
        lengths = df["business_name"].fillna("").astype(str).str.len()
        print("\nBusiness name length:")
        print(lengths.describe().to_string())

    if "business_address" in df.columns:
        lengths = df["business_address"].fillna("").astype(str).str.len()
        print("\nBusiness address length:")
        print(lengths.describe().to_string())

    if "entity_id" in df.columns:
        print("\nEntity ID:")
        print("  Duplicate IDs:", df["entity_id"].duplicated().sum())

    return df


def inspect_ground_truth(path):
    print("\n" + "=" * 80)
    print(path)
    print("=" * 80)

    gt = pd.read_csv(path, sep="\t")

    print(f"Rows: {len(gt):,}")
    print(f"Columns: {list(gt.columns)}")

    print("\nMissing:")
    print(gt.isna().sum().to_string())

    # Empty matched lists
    matches = gt["matched_entity_ids"].fillna("").astype(str).str.strip()

    empty = matches.eq("").sum()

    print(f"\nSingleton / no-match S1 entities: {empty:,}")
    print(f"Non-singletons:                  {len(gt) - empty:,}")
    print(f"Singleton percentage:             {empty / len(gt) * 100:.2f}%")

    # Number of matches per S1
    match_counts = matches.apply(
        lambda x: 0 if not x else len(x.split(","))
    )

    print("\nMatches per S1:")
    print(match_counts.describe(percentiles=[
        0.50, 0.75, 0.90, 0.95, 0.99, 0.999
    ]).to_string())

    print("\nMatch-count frequency:")
    print(match_counts.value_counts().sort_index().head(30).to_string())

    # Source-specific counts
    s2_counts = matches.apply(
        lambda x: sum(v.startswith("S2-") for v in x.split(",") if v)
    )

    s3_counts = matches.apply(
        lambda x: sum(v.startswith("S3-") for v in x.split(",") if v)
    )

    print("\nS2 matches:")
    print(s2_counts.describe().to_string())

    print("\nS3 matches:")
    print(s3_counts.describe().to_string())

    both = ((s2_counts > 0) & (s3_counts > 0)).sum()

    print(f"\nS1 entities matching BOTH S2 and S3: {both:,}")


print("\n\n######## TRAIN SOURCE FILES ########")

for name in [
    "train_source1.tsv",
    "train_source2.tsv",
    "train_source3.tsv",
]:
    inspect_file(ROOT / "train" / name)


inspect_ground_truth(ROOT / "train" / "train_ground_truth.tsv")


print("\n\n######## TEST SOURCE FILES ########")

for name in [
    "test_source1.tsv",
    "test_source2.tsv",
    "test_source3.tsv",
]:
    inspect_file(ROOT / "test" / name)


print("\n\nDONE")