import os
import yaml
import pandas as pd
from src.utils.logger import get_logger

logger = get_logger("data_loader")

def load_config(config_path: str = "config/config.yaml") -> dict:
    if not os.path.exists(config_path):
        return {
            "paths": {
                "raw_dir": ".",
                "trans_file": "trans.csv",
                "order_file": "order.csv",
                "account_file": "account.csv",
                "interim_dir": "data/interim",
                "processed_dir": "data/processed",
                "models_dir": "models",
                "recurring_model": "models/recurring_model.joblib",
                "anomaly_model": "models/anomaly_model.joblib"
            },
            "thresholds": {
                "recurring": {"min_txns": 3, "monthly_gap_min": 26, "monthly_gap_max": 35, "gap_cv_max": 0.25, "amt_cv_max": 0.20},
                "price_hike": {"min_pct_increase": 0.03},
                "duplicate": {"max_gap_hours": 24.0},
                "anomaly": {"contamination": 0.02}
            }
        }
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def load_raw_data(config: dict = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if config is None:
        config = load_config()
    
    raw_dir = config["paths"].get("raw_dir", ".")
    trans_path = os.path.join(raw_dir, config["paths"].get("trans_file", "trans.csv"))
    order_path = os.path.join(raw_dir, config["paths"].get("order_file", "order.csv"))
    account_path = os.path.join(raw_dir, config["paths"].get("account_file", "account.csv"))
    
    logger.info(f"Loading raw transactions from {trans_path}...")
    trans = pd.read_csv(trans_path, sep=";", low_memory=False)
    
    logger.info(f"Loading raw standing orders from {order_path}...")
    order = pd.read_csv(order_path, sep=";", low_memory=False)
    
    logger.info(f"Loading raw accounts from {account_path}...")
    account = pd.read_csv(account_path, sep=";", low_memory=False)
    
    logger.info(f"Loaded: trans={len(trans):,} rows, orders={len(order):,} rows, accounts={len(account):,} rows")
    return trans, order, account
