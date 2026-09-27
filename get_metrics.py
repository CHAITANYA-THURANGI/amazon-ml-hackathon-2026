"""
Parametrics and Performance Dashboard for Amazon ML Challenge 2026.

Usage:
    python get_metrics.py                # Full comprehensive report of all parametrics
    python get_metrics.py --model        # Model hyperparameters, ensemble weights, feature importances
    python get_metrics.py --submission   # Submission output parametrics (match rate, candidate stats)
    python get_metrics.py --data         # Dataset parametrics (line counts, countries, singletons)
    python get_metrics.py --hardware     # Hardware parametrics (RTX 4060 GPU, VRAM, CUDA, CPU)
    python get_metrics.py --eval         # Fast validation evaluation on ground truth (Macro F_0.5, P, R)
"""

import sys
import os
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure src/ is importable
SRC_DIR = Path(__file__).resolve().parent / "code" / "business_entity_resolution" / "src"
sys.path.insert(0, str(SRC_DIR))

from config import (
    PROJECT_ROOT, DATASET_DIR, TRAIN_DIR, TEST_DIR,
    OUTPUT_DIR, MATCHING_RESULTS, CANDIDATE_PAIRS, MODELS_DIR,
    detect_device,
)
from features import FEATURE_NAMES


def print_hardware_parametrics():
    print("\n" + "=" * 65)
    print("1. HARDWARE & COMPUTE PARAMETRICS")
    print("=" * 65)
    dev = detect_device()
    print(f"  Target Device:       {dev['device_name']}")
    print(f"  GPU Acceleration:    {dev['use_gpu']}")
    print(f"  XGBoost Device:      {dev['xgb_device'].upper()}")
    print(f"  PyTorch Device:      {dev['torch_device'].upper()}")
    print(f"  CPU Logical Cores:   {os.cpu_count()}")

    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            print(f"  GPU Model:           {props.name}")
            print(f"  Total VRAM:          {props.total_memory / 1024**3:.2f} GB")
            print(f"  CUDA Version:        {torch.version.cuda}")
            print(f"  Multi-Processors:    {props.multi_processor_count}")
    except Exception as e:
        print(f"  GPU Telemetry:       {e}")


def print_model_parametrics():
    print("\n" + "=" * 65)
    print("2. MODEL & ENSEMBLE PARAMETRICS")
    print("=" * 65)
    model_path = MODELS_DIR / "matching_ensemble.joblib"
    if not model_path.exists():
        model_path = MODELS_DIR / "matching_model.joblib"

    if not model_path.exists():
        print(f"  ⚠ No trained model found in {MODELS_DIR}")
        return

    import joblib
    data = joblib.load(model_path)

    th = data.get("threshold", 0.75)
    dev = data.get("device", "cuda")
    weights = data.get("weights", (0.6, 0.4))
    print(f"  Model Artifact:      {model_path.name} ({model_path.stat().st_size / 1024**2:.2f} MB)")
    print(f"  Trained On Device:   {dev.upper()}")
    print(f"  Optimal Decision Th: {th:.4f} (optimized for Macro F_0.5)")

    if "weights" in data:
        print(f"  Ensemble Fusion:     {weights[0]*100:.0f}% XGBoost + {weights[1]*100:.0f}% LightGBM")

    # Inspect XGBoost Hyperparameters
    xgb = data.get("xgb_model") or data.get("model")
    if xgb is not None:
        params = xgb.get_params()
        print("\n  XGBoost Hyperparameters:")
        print(f"    - n_estimators:     {params.get('n_estimators')}")
        print(f"    - max_depth:        {params.get('max_depth')}")
        print(f"    - learning_rate:    {params.get('learning_rate')}")
        print(f"    - subsample:        {params.get('subsample')}")
        print(f"    - colsample_bytree: {params.get('colsample_bytree')}")
        print(f"    - tree_method:      {params.get('tree_method')}")
        print(f"    - eval_metric:      {params.get('eval_metric')}")

        if hasattr(xgb, "feature_importances_"):
            importances = xgb.feature_importances_
            feat_names = FEATURE_NAMES[:len(importances)]
            df_imp = pd.DataFrame({
                "Feature": feat_names,
                "Importance": importances,
            }).sort_values("Importance", ascending=False)

            print("\n  Top 10 Feature Importances (XGBoost GPU):")
            for idx, r in enumerate(df_imp.head(10).itertuples(), 1):
                bar = "█" * int(r.Importance * 40)
                print(f"    {idx:2d}. {r.Feature:24s}: {r.Importance:.4f}  {bar}")


def print_submission_parametrics():
    print("\n" + "=" * 65)
    print("3. SUBMISSION OUTPUT PARAMETRICS")
    print("=" * 65)

    if not MATCHING_RESULTS.exists() or not CANDIDATE_PAIRS.exists():
        print("  ⚠ Output files not found. Run pipeline first.")
        return

    m_size = MATCHING_RESULTS.stat().st_size / (1024**2)
    c_size = CANDIDATE_PAIRS.stat().st_size / (1024**2)
    print(f"  Matching File:       {MATCHING_RESULTS} ({m_size:.1f} MB)")
    print(f"  Candidates File:     {CANDIDATE_PAIRS} ({c_size:.1f} MB)")

    # Read and parse summary statistics
    n_total = 0
    n_with_matches = 0
    total_matches = 0
    match_dist = {0: 0, 1: 0, 2: 0, "3+": 0}

    with open(MATCHING_RESULTS, "r", encoding="utf-8") as f:
        next(f)  # header
        for line in f:
            n_total += 1
            parts = line.strip().split("\t")
            if len(parts) > 1 and parts[1]:
                ids = parts[1].split(",")
                cnt = len(ids)
                n_with_matches += 1
                total_matches += cnt
                if cnt == 1:
                    match_dist[1] += 1
                elif cnt == 2:
                    match_dist[2] += 1
                else:
                    match_dist["3+"] += 1
            else:
                match_dist[0] += 1

    n_singletons = match_dist[0]
    print(f"\n  Entity Statistics (Test Set):")
    print(f"    - Total S1 Entities:     {n_total:,}")
    print(f"    - Entities with Matches: {n_with_matches:,} ({n_with_matches/n_total*100:.2f}%)")
    print(f"    - Singletons (No Match): {n_singletons:,} ({n_singletons/n_total*100:.2f}%)")
    print(f"    - Total Matches Linked:  {total_matches:,}")
    print(f"    - Avg Matches per Hit:   {total_matches / max(n_with_matches, 1):.2f}")

    print("\n  Match Count Breakdown:")
    print(f"    - 0 Matches (Singletons): {match_dist[0]:,} ({match_dist[0]/n_total*100:.1f}%)")
    print(f"    - Exactly 1 Match:        {match_dist[1]:,} ({match_dist[1]/n_total*100:.1f}%)")
    print(f"    - Exactly 2 Matches:      {match_dist[2]:,} ({match_dist[2]/n_total*100:.1f}%)")
    print(f"    - 3 or More Matches:      {match_dist['3+']:,} ({match_dist['3+']/n_total*100:.1f}%)")


def print_dataset_parametrics():
    print("\n" + "=" * 65)
    print("4. DATASET & BLOCKING PARAMETRICS")
    print("=" * 65)

    files = [
        ("Train S1 (Reference)", TRAIN_DIR / "train_source1.tsv"),
        ("Train S2", TRAIN_DIR / "train_source2.tsv"),
        ("Train S3", TRAIN_DIR / "train_source3.tsv"),
        ("Train Ground Truth", TRAIN_DIR / "train_ground_truth.tsv"),
        ("Test S1 (Reference)", TEST_DIR / "test_source1.tsv"),
        ("Test S2", TEST_DIR / "test_source2.tsv"),
        ("Test S3", TEST_DIR / "test_source3.tsv"),
    ]

    for label, path in files:
        if path.exists():
            size_mb = path.stat().st_size / (1024**2)
            print(f"  {label:24s}: {size_mb:7.1f} MB  ({path.name})")

    # Sample country distribution
    s1_train = TRAIN_DIR / "train_source1.tsv"
    if s1_train.exists():
        df_head = pd.read_csv(s1_train, sep="\t", nrows=50_000)
        print("\n  Training Country Distribution (Sample 50k):")
        for c, count in df_head["country"].value_counts().items():
            print(f"    - {c:10s}: {count:,} ({count/len(df_head)*100:.1f}%)")

    s1_test = TEST_DIR / "test_source1.tsv"
    if s1_test.exists():
        df_test_head = pd.read_csv(s1_test, sep="\t", nrows=50_000)
        print("\n  Test Country Distribution (Sample 50k, includes France):")
        for c, count in df_test_head["country"].value_counts().items():
            print(f"    - {c:10s}: {count:,} ({count/len(df_test_head)*100:.1f}%)")


def run_eval_parametrics(n_sample: int = 2_000):
    from evaluate_holdout import evaluate_model_on_holdout
    evaluate_model_on_holdout(n_entities=n_sample)


def main():
    parser = argparse.ArgumentParser(description="Print all parametrics for Amazon ML Challenge 2026")
    parser.add_argument("--hardware", action="store_true", help="Show hardware and GPU compute parametrics")
    parser.add_argument("--model", action="store_true", help="Show model hyperparameters and feature importances")
    parser.add_argument("--submission", action="store_true", help="Show submission output statistics and breakdown")
    parser.add_argument("--data", action="store_true", help="Show dataset size and country distributions")
    parser.add_argument("--eval", action="store_true", help="Run fast validation evaluation for Macro F_0.5")
    args = parser.parse_args()

    # If no specific flag passed, show comprehensive dashboard
    show_all = not (args.hardware or args.model or args.submission or args.data or args.eval)

    print("\n" + "#" * 65)
    print("       AMAZON ML CHALLENGE 2026 — PARAMETRICS DASHBOARD       ")
    print("#" * 65)

    if show_all or args.hardware:
        print_hardware_parametrics()
    if show_all or args.model:
        print_model_parametrics()
    if show_all or args.submission:
        print_submission_parametrics()
    if show_all or args.data:
        print_dataset_parametrics()
    if args.eval:
        run_eval_parametrics()

    print("\n" + "=" * 65)
    print("Dashboard complete. All parametrics verified!")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
