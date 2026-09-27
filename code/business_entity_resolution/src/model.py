"""
Matching Model module — GPU-accelerated XGBoost with automatic CPU fallback.

Optimizes decision threshold specifically for the competition metric (Macro F_0.5).
"""

import numpy as np
import pandas as pd
from pathlib import Path
import joblib

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

from sklearn.ensemble import HistGradientBoostingClassifier
from config import MODELS_DIR, RANDOM_SEED, DEFAULT_THRESHOLD, DEVICE_CONFIG


def compute_f05(precision: float, recall: float) -> float:
    """Compute F_0.5 score."""
    if precision + recall == 0:
        return 0.0
    return (1.25 * precision * recall) / (0.25 * precision + recall)


class EntityMatchingModel:
    """XGBoost entity matching classifier with GPU acceleration and CPU fallback."""

    def __init__(self, device: str = None, threshold: float = DEFAULT_THRESHOLD):
        self.threshold = threshold
        self.device = device or DEVICE_CONFIG["xgb_device"]
        self.model = None
        self._init_model()

    def _init_model(self):
        """Initialize XGBoost with GPU or fallback."""
        if HAS_XGBOOST:
            print(f"  Initializing XGBoost on device: {self.device.upper()}")
            self.model = xgb.XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.08,
                subsample=0.8,
                colsample_bytree=0.8,
                tree_method="hist",
                device=self.device,
                eval_metric="logloss",
                random_state=RANDOM_SEED,
                n_jobs=-1 if self.device == "cpu" else 1,
            )
        else:
            print("  XGBoost not found, using HistGradientBoostingClassifier (CPU)")
            self.model = HistGradientBoostingClassifier(
                max_iter=300,
                max_depth=6,
                learning_rate=0.08,
                random_state=RANDOM_SEED,
            )

    def train(self, X: np.ndarray, y: np.ndarray):
        """Train model with automatic device fallback if needed."""
        print(f"  Training on {len(X):,} samples ({y.sum():,} positive, {len(y)-y.sum():,} negative)...")
        try:
            self.model.fit(X, y)
        except Exception as e:
            if self.device == "cuda":
                print(f"  ⚠ GPU training encountered issue: {e}")
                print("  Falling back to CPU...")
                self.device = "cpu"
                self._init_model()
                self.model.fit(X, y)
            else:
                raise
        print("  Model training completed!")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict match probabilities."""
        if len(X) == 0:
            return np.array([])
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: np.ndarray, threshold: float = None) -> np.ndarray:
        """Predict binary matches based on threshold."""
        th = threshold if threshold is not None else self.threshold
        proba = self.predict_proba(X)
        return (proba >= th).astype(int)

    def tune_threshold(self, X_val: np.ndarray, y_val: np.ndarray) -> float:
        """Find the threshold that maximizes F_0.5 on validation data."""
        print("  Tuning decision threshold for F_0.5...")
        probas = self.predict_proba(X_val)
        best_th = self.threshold
        best_f05 = 0.0

        for th in np.arange(0.30, 0.90, 0.05):
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

        print(f"  Optimal threshold: {best_th:.2f} (Val F_0.5: {best_f05:.4f})")
        self.threshold = best_th
        return best_th

    def save(self, path: Path = None):
        """Save model to disk."""
        if path is None:
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            path = MODELS_DIR / "matching_model.joblib"
        joblib.dump({"model": self.model, "threshold": self.threshold, "device": self.device}, path)
        print(f"  Model saved to {path}")

    def load(self, path: Path = None):
        """Load model from disk."""
        if path is None:
            path = MODELS_DIR / "matching_model.joblib"
        data = joblib.load(path)
        self.model = data["model"]
        self.threshold = data.get("threshold", DEFAULT_THRESHOLD)
        self.device = data.get("device", "cpu")
        print(f"  Model loaded from {path}")
