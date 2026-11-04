import time
import pandas as pd
from src.models.predict import ExpenseDetector
from src.data.schema import TransactionInput
from src.utils.logger import get_logger

logger = get_logger("live_simulation")

def run_live_simulation(customer_id: str = "1"):
    print("\n" + "=" * 75)
    print("      LIVE TRANSACTION STREAMING & DETECTION SIMULATION")
    print(f"      Target Customer Account: {customer_id}")
    print("=" * 75 + "\n")
    
    detector = ExpenseDetector()
    
    # Load transactions for customer 1
    df = pd.read_csv("data/interim/clean_transactions.csv")
    cust_df = df[df["customer_id"].astype(str) == str(customer_id)].sort_values("date").head(15)
    
    print(f"1. Streaming {len(cust_df)} authentic ledger transactions...")
    active_sub_merchant = None
    for idx, row in cust_df.iterrows():
        t = TransactionInput(
            txn_id=str(row["txn_id"]),
            customer_id=str(row["customer_id"]),
            date=str(row["date"])[:10],
            merchant=str(row["merchant"]),
            amount=float(row["amount"]),
            category=str(row["category"]),
            payment_method=str(row["payment_method"]),
            currency="CZK"
        )
        res = detector.score_transaction(t)
        if res.is_recurring and active_sub_merchant is None:
            active_sub_merchant = t.merchant
        rec_status = "RECURRING" if res.is_recurring else "NORMAL"
        freq = f"({res.frequency})" if res.is_recurring else ""
        print(f"  [{t.date}] {t.merchant[:38]:<38} | {t.amount:>7.2f} CZK | {rec_status:<9} {freq:<9} | Prob: {res.recurring_probability:.2f}")
    
    print("\n" + "-" * 75)
    print("2. Simulating INJECTED EVENTS (Price Hike, Duplicate, Outlier Anomaly)...")
    print("-" * 75)
    
    # Event A: Injected Silent Price Hike on the active recurring subscription
    target_merchant = "Allianz Insurance (YZ) - Utilities & Household (#87144583)"
    hike_txn = TransactionInput(
        txn_id="sim_hike_901",
        customer_id=str(customer_id),
        date="1996-01-05",
        merchant=target_merchant,
        amount=2850.00,  # Increased from 2452.00 (+16.2%)
        category="Utilities & Household"
    )
    res_hike = detector.score_transaction(hike_txn)
    print(f"\n[EVENT A: PRICE HIKE TEST]")
    print(f"  Txn: {hike_txn.date} | {hike_txn.merchant} | {hike_txn.amount} CZK")
    print(f"  Is Price Hike Detected: {res_hike.is_price_increase}")
    for a in res_hike.alerts:
        print(f"  >> Alert: {a}")
        
    # Event B: Injected Duplicate Debit within same day
    dup_txn = TransactionInput(
        txn_id="sim_dup_902",
        customer_id=str(customer_id),
        date="1996-01-05",
        merchant=target_merchant,
        amount=2850.00,  # Identical amount within 24 hours
        category="Utilities & Household"
    )
    res_dup = detector.score_transaction(dup_txn)
    print(f"\n[EVENT B: DUPLICATE CHARGE TEST]")
    print(f"  Txn: {dup_txn.date} | {dup_txn.merchant} | {dup_txn.amount} CZK")
    print(f"  Is Duplicate Detected:  {res_dup.is_duplicate}")
    for a in res_dup.alerts:
        print(f"  >> Alert: {a}")
        
    # Event C: Injected Unusually High Outflow (Anomaly)
    ano_txn = TransactionInput(
        txn_id="sim_ano_903",
        customer_id=str(customer_id),
        date="1996-01-12",
        merchant="Luxury Jewelers Direct Debit",
        amount=89500.00,  # Extreme outlier
        category="General Outflow"
    )
    res_ano = detector.score_transaction(ano_txn)
    print(f"\n[EVENT C: OUTLIER ANOMALY TEST]")
    print(f"  Txn: {ano_txn.date} | {ano_txn.merchant} | {ano_txn.amount} CZK")
    print(f"  Is Outlier Detected:    {res_ano.is_anomaly} (Score: {res_ano.anomaly_score:.4f})")
    for a in res_ano.alerts:
        print(f"  >> Alert: {a}")

    # Summary
    print("\n" + "=" * 75)
    print("3. FINAL CUSTOMER INTELLIGENCE & SAVINGS REPORT")
    print("=" * 75)
    summary = detector.get_customer_summary(customer_id)
    print(f"Customer ID:                 {summary.customer_id}")
    print(f"Active Subscriptions:        {summary.active_subscriptions_count}")
    print(f"Total Monthly Subscription:  {summary.total_monthly_subscription_spend:,.2f} CZK")
    print(f"Projected Annual Outflow:    {summary.total_yearly_subscription_spend:,.2f} CZK")
    print(f"Actionable Annual Savings:   {summary.potential_savings:,.2f} CZK")
    print("\nActive Subscriptions Identified:")
    for s in summary.subscriptions:
        print(f"  * {s['merchant']}: {s['monthly_cost']:,.2f} CZK/mo [{s['frequency']}] (Next billing: {s['next_date']})")
    print("=" * 75 + "\n")

if __name__ == "__main__":
    run_live_simulation()
