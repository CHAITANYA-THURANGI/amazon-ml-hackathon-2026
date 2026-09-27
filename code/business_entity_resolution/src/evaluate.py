"""
Evaluation module — F₀.₅ score computation and validation split creation.

F₀.₅ is precision-heavy: weights precision 2× over recall.
Computed as a macro-average: F₀.₅ per S1 entity, then averaged.
"""

import numpy as np
import pandas as pd
from config import VAL_SPLIT_RATIO, RANDOM_SEED, SOURCE1_ENTITY_ID_COL, MATCHED_ENTITY_IDS_COL


def f_beta_score(precision: float, recall: float, beta: float = 0.5) -> float:
    """Compute F_beta score from precision and recall."""
    if precision + recall == 0:
        return 0.0
    beta_sq = beta ** 2
    return (1 + beta_sq) * precision * recall / (beta_sq * precision + recall)


def compute_entity_f05(predicted: set, actual: set) -> float:
    """
    Compute F₀.₅ for a single S1 entity.

    Singletons (actual is empty):
        - predict empty → 1.0
        - predict anything → 0.0
    """
    if len(actual) == 0:
        return 1.0 if len(predicted) == 0 else 0.0

    if len(predicted) == 0:
        return 0.0

    tp = len(predicted & actual)
    precision = tp / len(predicted) if len(predicted) > 0 else 0.0
    recall = tp / len(actual) if len(actual) > 0 else 0.0

    return f_beta_score(precision, recall, beta=0.5)


def evaluate_predictions(
    predictions: dict,  # {s1_id: set(matched_ids)}
    ground_truth: dict,  # {s1_id: set(matched_ids)}
) -> dict:
    """
    Evaluate predictions against ground truth using macro-averaged F₀.₅.

    Args:
        predictions: {s1_entity_id: set(matched_entity_ids)}
        ground_truth: {s1_entity_id: set(matched_entity_ids)}

    Returns:
        dict with 'f05_macro', 'precision_macro', 'recall_macro',
        'n_entities', 'n_singletons', 'n_correct_singletons'
    """
    scores = []
    precisions = []
    recalls = []
    n_singletons = 0
    n_correct_singletons = 0

    for s1_id, actual in ground_truth.items():
        predicted = predictions.get(s1_id, set())
        f05 = compute_entity_f05(predicted, actual)
        scores.append(f05)

        if len(actual) == 0:
            n_singletons += 1
            if len(predicted) == 0:
                n_correct_singletons += 1

        if len(predicted) > 0:
            tp = len(predicted & actual)
            precisions.append(tp / len(predicted))
        if len(actual) > 0:
            tp = len(predicted & actual)
            recalls.append(tp / len(actual))

    return {
        "f05_macro": np.mean(scores),
        "precision_macro": np.mean(precisions) if precisions else 0.0,
        "recall_macro": np.mean(recalls) if recalls else 0.0,
        "n_entities": len(ground_truth),
        "n_singletons": n_singletons,
        "n_correct_singletons": n_correct_singletons,
    }


def parse_ground_truth(gt_df: pd.DataFrame) -> dict:
    """
    Parse ground truth DataFrame into {s1_id: set(matched_ids)}.
    """
    result = {}
    for _, row in gt_df.iterrows():
        s1_id = row[SOURCE1_ENTITY_ID_COL]
        matched_str = str(row[MATCHED_ENTITY_IDS_COL]).strip()
        if matched_str and matched_str != "nan":
            matched = set(m.strip() for m in matched_str.split(",") if m.strip())
        else:
            matched = set()
        result[s1_id] = matched
    return result


def create_validation_split(
    gt_df: pd.DataFrame,
    val_ratio: float = VAL_SPLIT_RATIO,
    seed: int = RANDOM_SEED,
) -> tuple:
    """
    Split ground truth into train and validation sets.

    Returns:
        (train_gt_df, val_gt_df)
    """
    gt_shuffled = gt_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    split_idx = int(len(gt_shuffled) * (1 - val_ratio))
    train_gt = gt_shuffled.iloc[:split_idx]
    val_gt = gt_shuffled.iloc[split_idx:]
    print(f"  Train split: {len(train_gt):,} entities")
    print(f"  Val split:   {len(val_gt):,} entities")
    return train_gt, val_gt


def print_evaluation_report(results: dict):
    """Print a formatted evaluation report."""
    print("\n" + "=" * 50)
    print("EVALUATION REPORT")
    print("=" * 50)
    print(f"  F₀.₅ (macro):     {results['f05_macro']:.4f}")
    print(f"  Precision (macro): {results['precision_macro']:.4f}")
    print(f"  Recall (macro):    {results['recall_macro']:.4f}")
    print(f"  Total entities:    {results['n_entities']:,}")
    print(f"  Singletons:        {results['n_singletons']:,}")
    print(f"  Correct singletons:{results['n_correct_singletons']:,}")
    print("=" * 50)
