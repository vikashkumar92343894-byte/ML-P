import pandas as pd
import numpy as np

def label_recurring_rules(df: pd.DataFrame, min_txns: int = 3) -> pd.Series:
    """
    Heuristic rule engine for recurring subscription detection.
    Evaluates:
      1. Minimum transaction count threshold.
      2. Regular periodicity: monthly (~26-35 days), weekly (~6-8 days), or yearly (~355-375 days).
      3. Low dispersion: gap CV < 0.25 and amount CV < 0.20.
    """
    monthly = df["mean_gap"].between(26, 35)
    weekly = df["mean_gap"].between(6, 8)
    yearly = df["mean_gap"].between(355, 375)
    
    regular_cadence = (df["gap_cv"] <= 0.25) & (df["amt_cv"] <= 0.20)
    sufficient_history = df["n_txn"] >= min_txns
    
    return sufficient_history & (monthly | weekly | yearly) & regular_cadence

def get_cadence_frequency(mean_gap: float) -> str:
    """Returns human-readable cadence label."""
    if 26 <= mean_gap <= 35:
        return "Monthly"
    elif 6 <= mean_gap <= 8:
        return "Weekly"
    elif 355 <= mean_gap <= 375:
        return "Yearly"
    elif 85 <= mean_gap <= 95:
        return "Quarterly"
    else:
        return "Ad-hoc"

def detect_price_increases(df: pd.DataFrame, recurring_col: str = "is_recurring", min_pct: float = 0.03) -> pd.Series:
    """
    Flags price hikes where transaction amount increased by > 3% on an active recurring subscription.
    """
    is_rec = df[recurring_col].astype(bool)
    hike = is_rec & (df["pct_change"] >= min_pct) & (df["n_txn"] >= 2)
    return hike

def detect_duplicates(df: pd.DataFrame, max_hours: float = 24.0) -> pd.Series:
    """
    Flags potential duplicate charges if identical amount to same merchant appears within 24 hours.
    """
    if "hrs_since_same" not in df.columns:
        return pd.Series(False, index=df.index)
    return (df["hrs_since_same"].notnull()) & (df["hrs_since_same"] <= max_hours) & (df["hrs_since_same"] >= 0)

def detect_rule_anomalies(df: pd.DataFrame, z_thresh: float = 3.0) -> pd.Series:
    """
    Flags transactions where amount deviates significantly from the customer's typical expenditure.
    """
    ratio_spike = df["amt_to_mean_ratio"] >= z_thresh
    unusual_single = (df["n_txn"] == 1) & (df["amount"] > df["amount"].quantile(0.95))
    return ratio_spike | unusual_single
