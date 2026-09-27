"""
End-to-end pipeline — orchestrates all stages of entity resolution.

Stages:
1. Load & preprocess data
2. Candidate generation (blocking)
3. Feature extraction for candidate pairs
4. Model training (on training data with validation)
5. Inference on test candidates
6. Output generation (matching_results.tsv + candidate_pairs.tsv)
"""

import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Add src/ to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    TRAIN_SOURCE1, TRAIN_SOURCE2, TRAIN_SOURCE3, TRAIN_GROUND_TRUTH,
    TEST_SOURCE1, TEST_SOURCE2, TEST_SOURCE3,
    OUTPUT_DIR, MATCHING_RESULTS, CANDIDATE_PAIRS,
    TSV_SEP, ENTITY_ID_COL,
    SOURCE1_ENTITY_ID_COL, MATCHED_ENTITY_IDS_COL, CANDIDATE_ENTITY_IDS_COL,
    MATCH_THRESHOLD, TFIDF_TOP_K,
)
from preprocess import load_source, load_ground_truth
from blocking import generate_candidates
from features import extract_pair_features, get_feature_names
from model import MatchingModel
from evaluate import (
    parse_ground_truth, create_validation_split,
    evaluate_predictions, print_evaluation_report,
)


def build_training_data(
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    blocking_result: dict,
    ground_truth: dict,
) -> tuple:
    """
    Build feature matrix and labels from blocking candidates + ground truth.

    Returns:
        (X, y, pair_info) where pair_info = [(s1_id, cand_id), ...]
    """
    print("\n  Extracting features for training pairs...")

    # Index DataFrames by entity_id for fast lookup
    s2_index = s2_df.set_index(ENTITY_ID_COL).to_dict("index")
    s3_index = s3_df.set_index(ENTITY_ID_COL).to_dict("index")
    cand_index = {**s2_index, **s3_index}

    s1_index = s1_df.set_index(ENTITY_ID_COL).to_dict("index")

    feature_names = get_feature_names()
    X_rows = []
    y_labels = []
    pair_info = []

    for s1_id, candidates in tqdm(blocking_result.items(), desc="  Feature extraction"):
        s1_row = s1_index.get(s1_id)
        if s1_row is None:
            continue

        true_matches = ground_truth.get(s1_id, set())

        for cand_id in candidates:
            cand_row = cand_index.get(cand_id)
            if cand_row is None:
                continue

            features = extract_pair_features(s1_row, cand_row)
            X_rows.append([features[f] for f in feature_names])
            y_labels.append(1 if cand_id in true_matches else 0)
            pair_info.append((s1_id, cand_id))

    X = np.array(X_rows, dtype=np.float32)
    y = np.array(y_labels, dtype=np.int32)

    print(f"  Feature matrix: {X.shape}")
    print(f"  Positive pairs: {y.sum():,} ({y.mean():.2%})")
    print(f"  Negative pairs: {(1 - y).sum():,}")

    return X, y, pair_info


def predict_test(
    model: MatchingModel,
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    blocking_result: dict,
) -> dict:
    """
    Run inference on test candidates.

    Returns:
        dict: {s1_id: set(matched_entity_ids)}
    """
    print("\n  Running inference on test candidates...")

    # Index
    s2_index = s2_df.set_index(ENTITY_ID_COL).to_dict("index")
    s3_index = s3_df.set_index(ENTITY_ID_COL).to_dict("index")
    cand_index = {**s2_index, **s3_index}
    s1_index = s1_df.set_index(ENTITY_ID_COL).to_dict("index")

    feature_names = get_feature_names()
    predictions = {}

    for s1_id, candidates in tqdm(blocking_result.items(), desc="  Test inference"):
        s1_row = s1_index.get(s1_id)
        if s1_row is None or len(candidates) == 0:
            predictions[s1_id] = set()
            continue

        X_rows = []
        valid_cands = []

        for cand_id in candidates:
            cand_row = cand_index.get(cand_id)
            if cand_row is None:
                continue
            features = extract_pair_features(s1_row, cand_row)
            X_rows.append([features[f] for f in feature_names])
            valid_cands.append(cand_id)

        if not X_rows:
            predictions[s1_id] = set()
            continue

        X = np.array(X_rows, dtype=np.float32)
        preds = model.predict(X)
        matched = {valid_cands[i] for i, p in enumerate(preds) if p == 1}
        predictions[s1_id] = matched

    return predictions


def save_output(
    predictions: dict,
    blocking_result: dict,
    s1_ids: list,
):
    """Save matching_results.tsv and candidate_pairs.tsv."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # matching_results.tsv
    rows = []
    for s1_id in s1_ids:
        matched = predictions.get(s1_id, set())
        matched_str = ",".join(sorted(matched)) if matched else ""
        rows.append({SOURCE1_ENTITY_ID_COL: s1_id, MATCHED_ENTITY_IDS_COL: matched_str})

    df_match = pd.DataFrame(rows)
    df_match.to_csv(MATCHING_RESULTS, sep=TSV_SEP, index=False)
    print(f"\n  Saved: {MATCHING_RESULTS}")

    # candidate_pairs.tsv
    rows = []
    for s1_id in s1_ids:
        cands = blocking_result.get(s1_id, [])
        cand_str = ",".join(cands) if cands else ""
        rows.append({SOURCE1_ENTITY_ID_COL: s1_id, CANDIDATE_ENTITY_IDS_COL: cand_str})

    df_cand = pd.DataFrame(rows)
    df_cand.to_csv(CANDIDATE_PAIRS, sep=TSV_SEP, index=False)
    print(f"  Saved: {CANDIDATE_PAIRS}")


def main():
    """Run the full entity resolution pipeline."""
    start = time.time()
    print("=" * 60)
    print("BUSINESS ENTITY RESOLUTION PIPELINE")
    print("=" * 60)

    # ── Stage 1: Load & Preprocess ──
    print("\n[1/6] Loading and preprocessing data...")
    train_s1 = load_source(TRAIN_SOURCE1)
    train_s2 = load_source(TRAIN_SOURCE2)
    train_s3 = load_source(TRAIN_SOURCE3)
    gt_df = load_ground_truth(TRAIN_GROUND_TRUTH)
    print(f"  Train S1: {len(train_s1):,} | S2: {len(train_s2):,} | S3: {len(train_s3):,}")
    print(f"  Ground truth: {len(gt_df):,} entries")

    test_s1 = load_source(TEST_SOURCE1)
    test_s2 = load_source(TEST_SOURCE2)
    test_s3 = load_source(TEST_SOURCE3)
    print(f"  Test  S1: {len(test_s1):,} | S2: {len(test_s2):,} | S3: {len(test_s3):,}")

    # ── Stage 2: Validation Split ──
    print("\n[2/6] Creating validation split...")
    train_gt, val_gt = create_validation_split(gt_df)
    train_gt_dict = parse_ground_truth(train_gt)
    val_gt_dict = parse_ground_truth(val_gt)

    # ── Stage 3: Candidate Generation (Training) ──
    print("\n[3/6] Generating candidates for training S1 entities...")
    train_blocking = generate_candidates(train_s1, train_s2, train_s3, top_k=TFIDF_TOP_K)

    # ── Stage 4: Feature Extraction & Training ──
    print("\n[4/6] Building training data and training model...")
    full_gt = parse_ground_truth(gt_df)
    X_train, y_train, _ = build_training_data(
        train_s1, train_s2, train_s3, train_blocking, full_gt,
    )

    model = MatchingModel(model_type="xgboost", threshold=MATCH_THRESHOLD)
    model.train(X_train, y_train)

    # Print feature importance
    feature_names = get_feature_names()
    importance = model.feature_importance(feature_names)
    if not importance.empty:
        print("\n  Top 10 features:")
        print(importance.head(10).to_string(index=False))

    # ── Stage 5: Validation ──
    print("\n[5/6] Evaluating on validation set...")
    val_s1_ids = set(val_gt_dict.keys())
    val_blocking = {k: v for k, v in train_blocking.items() if k in val_s1_ids}
    val_predictions = predict_test(model, train_s1, train_s2, train_s3, val_blocking)
    results = evaluate_predictions(val_predictions, val_gt_dict)
    print_evaluation_report(results)

    # ── Stage 6: Test Inference & Output ──
    print("\n[6/6] Running on test set...")
    test_blocking = generate_candidates(test_s1, test_s2, test_s3, top_k=TFIDF_TOP_K)
    test_predictions = predict_test(model, test_s1, test_s2, test_s3, test_blocking)

    test_s1_ids = test_s1[ENTITY_ID_COL].tolist()
    save_output(test_predictions, test_blocking, test_s1_ids)

    elapsed = time.time() - start
    print(f"\n{'=' * 60}")
    print(f"Pipeline complete in {elapsed / 60:.1f} minutes")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
