import pandas as pd
import numpy as np
from src.utils.logger import get_logger
from src.features.merchant_normalizer import normalize_merchant

logger = get_logger("build_features")

def build_features(df: pd.DataFrame, keys: list = ["customer_id", "merchant"]) -> pd.DataFrame:
    """
    Builds group and transaction-level behavioral features for subscription & anomaly detection.
    """
    logger.info("Building features on transaction dataset...")
    df = df.copy()
    
    # Ensure date is datetime
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"])
    
    # 1. Sort strictly by keys + date
    df = df.sort_values(keys + ["date"]).reset_index(drop=True)
    
    # 2. Extract calendar signals
    df["day_of_month"] = df["date"].dt.day
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    
    # 3. Inter-transaction gaps per (customer_id, merchant)
    grp = df.groupby(keys)
    df["gap_days"] = grp["date"].diff().dt.total_seconds() / (24 * 3600)
    df["prev_amt"] = grp["amount"].shift(1)
    df["pct_change"] = ((df["amount"] - df["prev_amt"]) / df["prev_amt"].replace(0, np.nan)).fillna(0)
    
    # 4. Hours since identical transaction (customer, merchant, amount) for duplicate detection
    dup_grp = df.groupby(keys + ["amount"])
    df["hrs_since_same"] = dup_grp["date"].diff().dt.total_seconds() / 3600.0
    
    # 5. Group-level aggregations
    agg_df = grp.agg(
        n_txn=("amount", "size"),
        mean_amt=("amount", "mean"),
        std_amt=("amount", "std"),
        min_amt=("amount", "min"),
        max_amt=("amount", "max"),
        mean_gap=("gap_days", "mean"),
        std_gap=("gap_days", "std"),
        dom_std=("day_of_month", "std")
    ).reset_index()
    
    # Fill standard deviations for groups with < 2 elements
    agg_df["std_amt"] = agg_df["std_amt"].fillna(0)
    agg_df["std_gap"] = agg_df["std_gap"].fillna(0)
    agg_df["dom_std"] = agg_df["dom_std"].fillna(0)
    agg_df["mean_gap"] = agg_df["mean_gap"].fillna(0)
    
    # Coefficients of variation
    agg_df["amt_cv"] = (agg_df["std_amt"] / agg_df["mean_amt"].replace(0, np.nan)).fillna(0)
    agg_df["gap_cv"] = (agg_df["std_gap"] / agg_df["mean_gap"].replace(0, np.nan)).fillna(1.0)
    
    # Merge aggregations back onto individual transactions
    featured_df = pd.merge(df, agg_df, on=keys, how="left")
    
    # Relative transaction ratios
    featured_df["amt_to_mean_ratio"] = (featured_df["amount"] / featured_df["mean_amt"].replace(0, np.nan)).fillna(1.0)
    featured_df["gap_to_mean_ratio"] = (featured_df["gap_days"] / featured_df["mean_gap"].replace(0, np.nan)).fillna(1.0)
    
    # Gap cadence indicators
    featured_df["is_approx_monthly"] = featured_df["mean_gap"].between(26, 35).astype(int)
    featured_df["is_approx_weekly"] = featured_df["mean_gap"].between(6, 8).astype(int)
    featured_df["is_approx_yearly"] = featured_df["mean_gap"].between(355, 375).astype(int)
    
    logger.info(f"Feature engineering completed: {featured_df.shape[0]:,} rows, {featured_df.shape[1]} columns.")
    return featured_df

def run_feature_pipeline(
    input_path: str = "data/interim/clean_transactions.csv",
    output_path: str = "data/processed/features.csv"
):
    import os
    logger.info(f"Reading cleaned transactions from {input_path}...")
    df = pd.read_csv(input_path)
    featured_df = build_features(df)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    logger.info(f"Saving model-ready features to {output_path}...")
    featured_df.to_csv(output_path, index=False)
    logger.info("Feature engineering pipeline completed successfully.")
    return featured_df

if __name__ == "__main__":
    run_feature_pipeline()

