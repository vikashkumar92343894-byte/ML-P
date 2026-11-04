# Schema definitions
from typing import List, Optional
from pydantic import BaseModel, Field

class TransactionInput(BaseModel):
    txn_id: str
    customer_id: str
    date: str
    merchant: str
    amount: float
    txn_type: str = "DEBIT"
    category: str = "General"
    payment_method: str = "Transfer"
    currency: str = "CZK"

class ScoreResult(BaseModel):
    txn_id: str
    customer_id: str
    merchant: str
    amount: float
    date: str
    is_recurring: bool
    recurring_probability: float
    is_price_increase: bool
    is_duplicate: bool
    is_anomaly: bool
    anomaly_score: float
    frequency: str = "None"
    alerts: List[str] = Field(default_factory=list)

class CustomerSummary(BaseModel):
    customer_id: str
    total_monthly_subscription_spend: float
    total_yearly_subscription_spend: float
    active_subscriptions_count: int
    subscriptions: List[dict]
    recent_alerts: List[str]
    potential_savings: float
