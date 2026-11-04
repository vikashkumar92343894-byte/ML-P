import os
import time
import joblib
import pytest
import numpy as np
import pandas as pd
from src.models.predict import ExpenseDetector
from src.data.schema import TransactionInput

@pytest.fixture
def detector():
    return ExpenseDetector(
        recurring_model_path="models/recurring_model.joblib",
        anomaly_model_path="models/anomaly_model.joblib"
    )

def test_model_artifacts_exist():
    assert os.path.exists("models/recurring_model.joblib"), "Recurring model artifact missing"
    assert os.path.exists("models/anomaly_model.joblib"), "Anomaly model artifact missing"

def test_model_load_and_predict(detector):
    assert detector.recurring_model is not None
    assert detector.anomaly_model is not None
    assert len(detector.recurring_cols) > 0
    assert len(detector.anomaly_cols) > 0

def test_inference_latency_sla(detector):
    """Verifies that scoring a single transaction takes well under the 200ms blueprint SLA."""
    t = TransactionInput(
        txn_id="sla_001",
        customer_id="cust_test_sla",
        date="1997-01-01",
        merchant="Spotify Premium",
        amount=149.0
    )
    
    start_time = time.perf_counter()
    res = detector.score_transaction(t)
    duration_ms = (time.perf_counter() - start_time) * 1000.0
    
    assert duration_ms < 200.0, f"Latency SLA violated: took {duration_ms:.2f} ms"
    assert res.txn_id == "sla_001"

def test_recurring_sequence_detection(detector):
    """Tests the transition from unseen merchant to confirmed recurring monthly subscription."""
    cust_id = "test_seq_user"
    dates = ["1996-01-10", "1996-02-10", "1996-03-10", "1996-04-10"]
    merchant = "O2 Czech Telecom Services"
    amount = 599.0
    
    results = []
    for i, d in enumerate(dates):
        t = TransactionInput(
            txn_id=f"seq_{i}",
            customer_id=cust_id,
            date=d,
            merchant=merchant,
            amount=amount
        )
        results.append(detector.score_transaction(t))
    
    # 1st payment: not yet recurring
    assert results[0].is_recurring is False
    # Later payments: recognized as monthly recurring with high probability
    assert results[-1].is_recurring is True
    assert results[-1].frequency == "Monthly"
    assert results[-1].recurring_probability >= 0.80

def test_anomaly_detection_spike(detector):
    """Tests that an unprecedented 50x spike in amount triggers the anomaly detection layer."""
    t_normal = TransactionInput(
        txn_id="ano_norm",
        customer_id="cust_ano_user",
        date="1996-05-01",
        merchant="Corner Grocery Store",
        amount=150.0
    )
    detector.score_transaction(t_normal)
    
    t_spike = TransactionInput(
        txn_id="ano_spike",
        customer_id="cust_ano_user",
        date="1996-05-15",
        merchant="Corner Grocery Store",
        amount=45000.0  # 300x spike
    )
    res_spike = detector.score_transaction(t_spike)
    assert res_spike.is_anomaly is True
    assert any("UNUSUAL" in a for a in res_spike.alerts)
