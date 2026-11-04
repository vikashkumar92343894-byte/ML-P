import os
import pytest
import pandas as pd
from src.api.main import score_batch, get_customer_summary, reset_history
from src.data.schema import TransactionInput

REQUIRED_SCHEMA_COLS = [
    "txn_id", "customer_id", "date", "merchant", "amount",
    "txn_type", "category", "payment_method", "currency",
    "is_ground_truth_recurring"
]

def test_clean_data_schema_contract():
    data_path = "data/interim/clean_transactions.csv"
    assert os.path.exists(data_path), "clean_transactions.csv missing"
    
    # Check header and types on first 1000 rows
    df = pd.read_csv(data_path, nrows=1000)
    for col in REQUIRED_SCHEMA_COLS:
        assert col in df.columns, f"Required column {col} missing from clean data schema"
    
    assert df["amount"].isnull().sum() == 0, "Null amounts detected"
    assert df["txn_id"].isnull().sum() == 0, "Null txn_ids detected"
    assert df["customer_id"].isnull().sum() == 0, "Null customer_ids detected"
    assert (df["amount"] < 0).sum() == 0, "Negative debit amounts found"

def test_batch_scoring_pipeline():
    reset_history()
    batch_txns = [
        TransactionInput(
            txn_id=f"batch_{i}",
            customer_id="cust_batch_test",
            date=f"1997-0{i+1}-05",
            merchant="Prague Energy Utility (AB)",
            amount=1500.0,
            category="Utilities & Household"
        )
        for i in range(4)
    ]
    
    results = score_batch(batch_txns)
    assert len(results) == 4
    assert results[-1].is_recurring is True
    assert results[-1].frequency == "Monthly"
    
    summary = get_customer_summary("cust_batch_test")
    assert summary.active_subscriptions_count >= 1
    assert summary.total_monthly_subscription_spend == 1500.0
