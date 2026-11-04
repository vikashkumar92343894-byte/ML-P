import pytest
from src.api.main import health_check, score_transaction, get_customer_summary, reset_history
from src.data.schema import TransactionInput

def test_health_endpoint():
    res = health_check()
    assert res["status"] == "healthy"
    assert "active_groups_in_memory" in res

def test_score_and_duplicate_alert():
    reset_history()
    
    # 1. First transaction
    t1 = TransactionInput(
        txn_id="tx_101",
        customer_id="cust_1",
        date="1996-05-01",
        merchant="Netflix Streaming",
        amount=299.0
    )
    res1 = score_transaction(t1)
    assert res1.is_duplicate is False
    
    # 2. Duplicate transaction within 24 hours
    t2 = TransactionInput(
        txn_id="tx_102",
        customer_id="cust_1",
        date="1996-05-01",
        merchant="Netflix Streaming",
        amount=299.0
    )
    res2 = score_transaction(t2)
    assert res2.is_duplicate is True
    assert any("DUPLICATE" in a for a in res2.alerts)

def test_customer_summary():
    summary = get_customer_summary("cust_1")
    assert summary.customer_id == "cust_1"
    assert hasattr(summary, "subscriptions")
    assert hasattr(summary, "total_monthly_subscription_spend")
