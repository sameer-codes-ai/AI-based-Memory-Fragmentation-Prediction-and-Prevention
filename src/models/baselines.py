"""
PREFRAG: Baseline Models
Implements B1 (Static Threshold), B2 (Moving Average), B3 (Linear Regression),
B4 (Random Forest), and B5 (Gradient Boosted Trees).
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler


class StaticThresholdBaseline:
    """
    B1: Static current-state heuristic. Assumes future fragmentation equals current fragmentation.
    """

    def __init__(self, threshold: float = 0.65):
        self.threshold = threshold

    def predict(self, X: np.ndarray) -> np.ndarray:
        # Index 14 is current_f1
        current_f1 = X[:, 14:15]
        # Multi-horizon prediction is constant current state
        return np.repeat(current_f1, 4, axis=1)

    def predict_risk(self, X: np.ndarray) -> np.ndarray:
        current_f1 = X[:, 14]
        return (current_f1 >= self.threshold).astype(int)


class MovingAverageBaseline:
    """
    B2: Moving average of recent fragmentation values.
    """

    def __init__(self, window: int = 5, threshold: float = 0.65):
        self.window = window
        self.threshold = threshold

    def predict(self, X: np.ndarray) -> np.ndarray:
        # Uses current_f1 and delta_f1 (index 14 and 20) to compute moving trend
        current_f1 = X[:, 14:15]
        delta_f1 = X[:, 20:21]
        
        t1 = current_f1 + (delta_f1 * 0.2)
        t5 = current_f1 + (delta_f1 * 0.5)
        t10 = current_f1 + (delta_f1 * 1.0)
        t30 = current_f1 + (delta_f1 * 1.5)
        preds = np.clip(np.hstack([t1, t5, t10, t30]), 0.0, 1.0)
        return preds

    def predict_risk(self, X: np.ndarray) -> np.ndarray:
        preds = self.predict(X)
        return (np.max(preds[:, :3], axis=1) >= self.threshold).astype(int)


class LinearRegressionBaseline:
    """
    B3: Regularized Ridge Linear Regression multi-horizon model.
    """

    def __init__(self, alpha: float = 1.0):
        self.scaler = StandardScaler()
        self.model = MultiOutputRegressor(Ridge(alpha=alpha, random_state=42))
        self.is_fitted = False

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "LinearRegressionBaseline":
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, Y)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted")
        X_scaled = self.scaler.transform(X)
        preds = self.model.predict(X_scaled)
        return np.clip(preds, 0.0, 1.0)


class RandomForestBaseline:
    """
    B4: Non-linear Random Forest multi-horizon regressor.
    """

    def __init__(self, n_estimators: int = 100, max_depth: int = 12):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1
        )
        self.is_fitted = False

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "RandomForestBaseline":
        self.model.fit(X, Y)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted")
        preds = self.model.predict(X)
        return np.clip(preds, 0.0, 1.0)

    def get_feature_importances(self) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted")
        return self.model.feature_importances_


class GradientBoostingBaseline:
    """
    B5: Multi-output Gradient Boosted Decision Trees baseline.
    """

    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.05, max_depth: int = 5):
        self.model = MultiOutputRegressor(GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=42
        ))
        self.is_fitted = False

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "GradientBoostingBaseline":
        self.model.fit(X, Y)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted")
        preds = self.model.predict(X)
        return np.clip(preds, 0.0, 1.0)
