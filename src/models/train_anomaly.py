import os
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from src.utils.logger import get_logger
from src.features.build_features import build_features

logger = get_logger("train_anomaly")

ANOMALY_FEATURE_COLS = [
    "amount",
    "mean_amt",
    "amt_to_mean_ratio",
    "gap_days",
    "gap_to_mean_ratio"
]

def train_anomaly_model(
    data_path: str = "data/interim/clean_transactions.csv",
    model_output_path: str = "models/anomaly_model.joblib",
    contamination: float = 0.02
) -> dict:
    logger.info(f"Loading data for anomaly training from {data_path}...")
    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"])
    
    featured_df = build_features(df)
    X = featured_df[ANOMALY_FEATURE_COLS].fillna(0)
    
    logger.info(f"Training IsolationForest with contamination={contamination} on {len(X):,} samples...")
    iso = IsolationForest(
        contamination=contamination,
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    iso.fit(X)
    
    # Calculate anomaly scores (-score_samples so higher means more anomalous)
    scores = -iso.score_samples(X)
    threshold = np.percentile(scores, (1 - contamination) * 100)
    logger.info(f"Anomaly score 98th percentile threshold: {threshold:.4f}")
    
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump({
        "model": iso,
        "feature_cols": ANOMALY_FEATURE_COLS,
        "threshold": threshold
    }, model_output_path)
    logger.info(f"Saved anomaly model artifact to {model_output_path}")
    
    return {"threshold": threshold, "n_samples": len(X)}

if __name__ == "__main__":
    train_anomaly_model()
