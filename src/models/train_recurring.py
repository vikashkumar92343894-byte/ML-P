import os
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from src.utils.logger import get_logger
from src.data.loader import load_config
from src.features.build_features import build_features
from src.labeling.rules import label_recurring_rules

logger = get_logger("train_recurring")

FEATURE_COLS = [
    "amount",
    "n_txn",
    "mean_amt",
    "std_amt",
    "amt_cv",
    "mean_gap",
    "std_gap",
    "gap_cv",
    "dom_std",
    "amt_to_mean_ratio",
    "gap_to_mean_ratio",
    "is_approx_monthly",
    "is_approx_weekly",
    "is_approx_yearly"
]

def train_recurring_model(
    data_path: str = "data/interim/clean_transactions.csv",
    split_date: str = "1997-07-01",
    model_output_path: str = "models/recurring_model.joblib"
) -> dict:
    logger.info(f"Loading cleaned transactions from {data_path}...")
    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"])
    
    # Generate features
    featured_df = build_features(df)
    
    # Time-based train / test split strictly avoiding lookahead bias
    logger.info(f"Applying time-based split at {split_date}...")
    train_mask = featured_df["date"] < split_date
    test_mask = featured_df["date"] >= split_date
    
    X_train = featured_df.loc[train_mask, FEATURE_COLS].fillna(0)
    y_train = featured_df.loc[train_mask, "is_ground_truth_recurring"]
    
    X_test = featured_df.loc[test_mask, FEATURE_COLS].fillna(0)
    y_test = featured_df.loc[test_mask, "is_ground_truth_recurring"]
    
    logger.info(f"Train split: {len(X_train):,} rows ({y_train.mean():.2%} recurring)")
    logger.info(f"Test split: {len(X_test):,} rows ({y_test.mean():.2%} recurring)")
    
    # 1. Rule Baseline Evaluation
    logger.info("Evaluating Rule-based Baseline on Test Set...")
    test_rules = label_recurring_rules(featured_df.loc[test_mask])
    rule_p = precision_score(y_test, test_rules, zero_division=0)
    rule_r = recall_score(y_test, test_rules, zero_division=0)
    rule_f1 = f1_score(y_test, test_rules, zero_division=0)
    logger.info(f"Rule Baseline: Precision={rule_p:.4f}, Recall={rule_r:.4f}, F1={rule_f1:.4f}")
    
    # 2. Train Random Forest Classifier
    logger.info("Training Random Forest Classifier with class_weight='balanced'...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        min_samples_split=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # 3. Model Evaluation on Test Set
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    ml_p = precision_score(y_test, y_pred, zero_division=0)
    ml_r = recall_score(y_test, y_pred, zero_division=0)
    ml_f1 = f1_score(y_test, y_pred, zero_division=0)
    ml_roc_auc = roc_auc_score(y_test, y_prob)
    ml_pr_auc = average_precision_score(y_test, y_prob)
    
    logger.info("=" * 50)
    logger.info(f"ML Model Results: Precision={ml_p:.4f}, Recall={ml_r:.4f}, F1={ml_f1:.4f}")
    logger.info(f"ROC-AUC={ml_roc_auc:.4f}, PR-AUC={ml_pr_auc:.4f}")
    logger.info("=" * 50)
    
    # Save artifacts
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump({
        "model": model,
        "feature_cols": FEATURE_COLS,
        "metrics": {
            "ml_precision": ml_p,
            "ml_recall": ml_r,
            "ml_f1": ml_f1,
            "ml_pr_auc": ml_pr_auc,
            "rule_f1": rule_f1
        }
    }, model_output_path)
    logger.info(f"Saved recurring model artifact to {model_output_path}")
    
    return {
        "rule_f1": rule_f1,
        "ml_f1": ml_f1,
        "ml_precision": ml_p,
        "ml_recall": ml_r,
        "ml_pr_auc": ml_pr_auc
    }

if __name__ == "__main__":
    train_recurring_model()
