"""
Futuristic Entity Matching Ensemble Model: GPU XGBoost + LightGBM Blending.

Combines:
1. GPU-accelerated XGBoost (depth-wise tree expansion on CUDA)
2. LightGBM (leaf-wise tree expansion with histogram binning)
3. Precision-calibrated Macro F_0.5 threshold optimization
"""

import numpy as np
import pandas as pd
from pathlib import Path
import joblib

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

from sklearn.ensemble import HistGradientBoostingClassifier
from config import MODELS_DIR, RANDOM_SEED, DEFAULT_THRESHOLD, DEVICE_CONFIG


def compute_f05(precision: float, recall: float) -> float:
    """Compute F_0.5 score."""
    if precision + recall == 0:
        return 0.0
    return (1.25 * precision * recall) / (0.25 * precision + recall)


class EntityMatchingEnsemble:
    """High-performance dual ensemble (XGBoost GPU + LightGBM)."""

    def __init__(self, device: str = None, threshold: float = DEFAULT_THRESHOLD):
        self.threshold = threshold
        self.device = device or DEVICE_CONFIG["xgb_device"]
        self.xgb_model = None
        self.lgb_model = None
        self.weights = (0.6, 0.4)  # (XGBoost, LightGBM)
        self._init_models()

    def _init_models(self):
        """Initialize both ensemble components."""
        # 1. XGBoost Model (GPU-accelerated)
        if HAS_XGB:
            print(f"  Initializing XGBoost on device: {self.device.upper()}", flush=True)
            self.xgb_model = xgb.XGBClassifier(
                n_estimators=400,
                max_depth=7,
                learning_rate=0.07,
                subsample=0.85,
                colsample_bytree=0.85,
                tree_method="hist",
                device=self.device,
                eval_metric="logloss",
                random_state=RANDOM_SEED,
                n_jobs=-1 if self.device == "cpu" else 1,
            )
        else:
            self.xgb_model = HistGradientBoostingClassifier(
                max_iter=300, max_depth=7, learning_rate=0.07, random_state=RANDOM_SEED
            )

        # 2. LightGBM Model
        if HAS_LGB:
            print("  Initializing LightGBM (leaf-wise gradient boosting)", flush=True)
            self.lgb_model = lgb.LGBMClassifier(
                n_estimators=400,
                num_leaves=63,
                learning_rate=0.07,
                subsample=0.85,
                colsample_bytree=0.85,
                random_state=RANDOM_SEED,
                n_jobs=-1,
                verbosity=-1,
            )
        else:
            self.lgb_model = None

    def train(self, X: np.ndarray, y: np.ndarray):
        """Train both models in the ensemble."""
        pos = int(y.sum())
        neg = len(y) - pos
        print(f"  Training Ensemble on {len(X):,} candidate pairs ({pos:,} positive, {neg:,} negative)...", flush=True)

        # Train XGBoost
        print("  --> Fitting XGBoost (CUDA GPU)...", flush=True)
        try:
            self.xgb_model.fit(X, y)
        except Exception as e:
            if self.device == "cuda":
                print(f"  ⚠ XGBoost GPU encountered: {e}. Falling back to CPU...", flush=True)
                self.device = "cpu"
                self.xgb_model.set_params(device="cpu")
                self.xgb_model.fit(X, y)
            else:
                raise

        # Train LightGBM
        if self.lgb_model is not None:
            print("  --> Fitting LightGBM...", flush=True)
            self.lgb_model.fit(X, y)

        print("  Ensemble training complete!", flush=True)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Compute weighted blended probability from the ensemble."""
        if len(X) == 0:
            return np.array([])

        p_xgb = self.xgb_model.predict_proba(X)[:, 1]

        if self.lgb_model is not None:
            p_lgb = self.lgb_model.predict_proba(X)[:, 1]
            w_xgb, w_lgb = self.weights
            return (w_xgb * p_xgb) + (w_lgb * p_lgb)

        return p_xgb

    def predict(self, X: np.ndarray, threshold: float = None) -> np.ndarray:
        """Predict binary matches based on precision-calibrated threshold."""
        th = threshold if threshold is not None else self.threshold
        proba = self.predict_proba(X)
        return (proba >= th).astype(int)

    def tune_threshold(self, X_val: np.ndarray, y_val: np.ndarray, entity_groups: list = None) -> float:
        """Optimize threshold specifically to maximize Macro F_0.5."""
        print("  Precision-calibrating decision threshold for Macro F_0.5...", flush=True)
        probas = self.predict_proba(X_val)
        best_th = 0.55
        best_f05 = 0.0

        if entity_groups is not None:
            # Group pairs by entity for true Macro F_0.5 computation
            from collections import defaultdict
            entity_cand_map = defaultdict(list)
            for idx, eid in enumerate(entity_groups):
                entity_cand_map[eid].append((probas[idx], y_val[idx]))

            for th in np.arange(0.40, 0.85, 0.025):
                scores = []
                for eid, items in entity_cand_map.items():
                    actual_pos = sum(1 for _, y in items if y == 1)
                    pred_pos = sum(1 for p, _ in items if p >= th)
                    tp = sum(1 for p, y in items if p >= th and y == 1)

                    if actual_pos == 0:
                        scores.append(1.0 if pred_pos == 0 else 0.0)
                    else:
                        p = tp / pred_pos if pred_pos > 0 else 0.0
                        r = tp / actual_pos if actual_pos > 0 else 0.0
                        scores.append(compute_f05(p, r))

                m_f05 = float(np.mean(scores))
                if m_f05 > best_f05:
                    best_f05 = m_f05
                    best_th = float(th)
        else:
            # Fallback to balanced F_0.5 search
            for th in np.arange(0.45, 0.80, 0.025):
                preds = (probas >= th).astype(int)
                tp = int(((preds == 1) & (y_val == 1)).sum())
                fp = int(((preds == 1) & (y_val == 0)).sum())
                fn = int(((preds == 0) & (y_val == 1)).sum())
                prec = tp / max(tp + fp, 1)
                rec = tp / max(tp + fn, 1)
                f05 = compute_f05(prec, rec)
                if f05 > best_f05:
                    best_f05 = f05
                    best_th = float(th)

        print(f"  ★ Optimal Decision Threshold: {best_th:.4f} (Validation Macro F_0.5: {best_f05:.4f})", flush=True)
        self.threshold = best_th
        return best_th

    def save(self, path: Path = None):
        """Save ensemble state to disk."""
        if path is None:
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            path = MODELS_DIR / "matching_ensemble.joblib"
        state = {
            "xgb_model": self.xgb_model,
            "lgb_model": self.lgb_model,
            "weights": self.weights,
            "threshold": self.threshold,
            "device": self.device,
        }
        joblib.dump(state, path)
        print(f"  Ensemble saved to {path}", flush=True)

    def load(self, path: Path = None):
        """Load ensemble state from disk."""
        if path is None:
            path = MODELS_DIR / "matching_ensemble.joblib"
        state = joblib.load(path)
        self.xgb_model = state["xgb_model"]
        self.lgb_model = state.get("lgb_model")
        self.weights = state.get("weights", (0.6, 0.4))
        self.threshold = state.get("threshold", DEFAULT_THRESHOLD)
        self.device = state.get("device", "cpu")
        print(f"  Ensemble loaded from {path}", flush=True)
