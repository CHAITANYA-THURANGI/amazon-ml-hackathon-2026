"""
Matching model — training and inference for entity resolution.

Uses a binary classifier (XGBoost by default) to score candidate pairs
and determine which ones are true matches.
"""

import numpy as np
import pandas as pd
from pathlib import Path

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

from sklearn.ensemble import GradientBoostingClassifier
from config import MATCH_THRESHOLD, MODELS_DIR, RANDOM_SEED


class MatchingModel:
    """Binary classifier for entity matching."""

    def __init__(self, model_type: str = "xgboost", threshold: float = MATCH_THRESHOLD):
        self.model_type = model_type
        self.threshold = threshold
        self.model = None
        self._build_model()

    def _build_model(self):
        """Initialize the underlying model."""
        if self.model_type == "xgboost" and HAS_XGBOOST:
            self.model = xgb.XGBClassifier(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=1.0,  # Adjust based on class imbalance
                eval_metric="logloss",
                random_state=RANDOM_SEED,
                n_jobs=-1,
            )
        elif self.model_type == "lightgbm" and HAS_LIGHTGBM:
            self.model = lgb.LGBMClassifier(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=RANDOM_SEED,
                n_jobs=-1,
                verbose=-1,
            )
        else:
            # Fallback to sklearn
            print(f"  Using sklearn GradientBoosting (install xgboost/lightgbm for better performance)")
            self.model = GradientBoostingClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                random_state=RANDOM_SEED,
            )

    def train(self, X: np.ndarray, y: np.ndarray):
        """
        Train the matching model.

        Args:
            X: Feature matrix (n_pairs, n_features)
            y: Binary labels (1 = match, 0 = non-match)
        """
        print(f"  Training {self.model_type} model...")
        print(f"  Samples: {len(y):,} | Positive: {y.sum():,} ({y.mean():.2%})")
        self.model.fit(X, y)
        print("  Training complete.")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return match probabilities for candidate pairs."""
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return binary match predictions based on threshold."""
        proba = self.predict_proba(X)
        return (proba >= self.threshold).astype(int)

    def feature_importance(self, feature_names: list) -> pd.DataFrame:
        """Return feature importances sorted descending."""
        if hasattr(self.model, "feature_importances_"):
            imp = self.model.feature_importances_
        else:
            return pd.DataFrame()

        df = pd.DataFrame({
            "feature": feature_names,
            "importance": imp,
        }).sort_values("importance", ascending=False)

        return df

    def save(self, path: Path = None):
        """Save model to disk."""
        if path is None:
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            path = MODELS_DIR / f"matching_model_{self.model_type}.pkl"

        import joblib
        joblib.dump(self.model, path)
        print(f"  Model saved to {path}")

    def load(self, path: Path = None):
        """Load model from disk."""
        if path is None:
            path = MODELS_DIR / f"matching_model_{self.model_type}.pkl"

        import joblib
        self.model = joblib.load(path)
        print(f"  Model loaded from {path}")
