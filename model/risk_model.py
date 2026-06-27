"""
Rockfall risk-scoring model.

Two layers, matching the "start simple, then upgrade" approach from the
build plan:

1. RuleBasedScorer   - transparent threshold/weighted-sum baseline.
2. MLRiskScorer      - RandomForestRegressor trained on engineered features
                       from the synthetic sensor data, predicting a 0-100
                       risk score. Falls back to the rule-based scorer if
                       no trained model is available yet.

Both scorers expose the same interface: score(features: dict) -> float
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

MODEL_PATH = os.path.join(os.path.dirname(__file__), "risk_model.joblib")


class RuleBasedScorer:
    """Transparent weighted-threshold baseline. Good fallback / explainability."""

    WEIGHTS = {
        "rainfall_mm_hr": 4.0,
        "vibration_mm_s": 6.0,
        "displacement_rate_mm_hr": 10.0,
        "pore_pressure_kpa": 0.3,
    }

    def score(self, features: dict) -> float:
        raw = sum(self.WEIGHTS[k] * features.get(k, 0) for k in self.WEIGHTS)
        return float(min(100, max(0, raw)))


class MLRiskScorer:
    def __init__(self):
        self.model = None
        if os.path.exists(MODEL_PATH):
            self.model = joblib.load(MODEL_PATH)
        self.fallback = RuleBasedScorer()

    FEATURE_ORDER = ["rainfall_mm_hr", "vibration_mm_s", "displacement_rate_mm_hr", "pore_pressure_kpa"]

    def score(self, features: dict) -> float:
        if self.model is None:
            return self.fallback.score(features)
        x = np.array([[features.get(f, 0) for f in self.FEATURE_ORDER]])
        pred = self.model.predict(x)[0]
        return float(min(100, max(0, pred)))


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add a displacement-rate feature (mm/hr) per zone from cumulative displacement."""
    df = df.sort_values(["zone", "timestamp"]).copy()
    df["displacement_rate_mm_hr"] = (
        df.groupby("zone")["displacement_mm"].diff().fillna(0) * 12
    )
    return df


def make_pseudo_labels(df: pd.DataFrame) -> pd.Series:
    """
    For training a baseline model without real incident labels, derive a
    pseudo risk-score target from the same signals using a slightly noisy
    rule, so the ML model learns a smoothed, non-linear version of it.
    Replace this with real historical incident outcomes when available.
    """
    scorer = RuleBasedScorer()
    noise = np.random.normal(0, 3, size=len(df))
    labels = df.apply(lambda r: scorer.score(r.to_dict()), axis=1) + noise
    return labels.clip(0, 100)


def train(csv_path: str = "../data/sensor_data.csv"):
    df = pd.read_csv(csv_path)
    df = engineer_features(df)
    y = make_pseudo_labels(df)
    X = df[MLRiskScorer.FEATURE_ORDER]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=150, max_depth=8, random_state=42)
    model.fit(X_train, y_train)

    score = model.score(X_test, y_test)
    print(f"Validation R^2: {score:.3f}")

    joblib.dump(model, MODEL_PATH)
    print(f"Model saved -> {MODEL_PATH}")


def tier_for_score(score: float) -> str:
    if score >= 70:
        return "EVACUATE"
    if score >= 40:
        return "WARNING"
    return "ADVISORY"


if __name__ == "__main__":
    train()
