import time
import requests
import pandas as pd
from typing import Optional
from src.utils.logger import get_logger

logger = get_logger("replay_streamer")

def replay_stream(
    csv_path: str = "data/interim/clean_transactions.csv",
    api_url: str = "http://localhost:8000/score",
    customer_id: Optional[str] = "1",
    delay_sec: float = 0.02,
    max_records: int = 30,
    inject_events: bool = True
):
    """
    Simulates real-time transaction ingestion by streaming records sequentially into the live FastAPI microservice.
    """
    base_url = api_url.rsplit("/", 1)[0]
    
    # 1. Reset API state for a clean streaming demo
    try:
        reset_resp = requests.post(f"{base_url}/reset", timeout=5)
        logger.info(f"Connected to API ({base_url}). History reset: {reset_resp.json().get('message')}")
    except Exception as e:
        logger.error(f"Could not connect to FastAPI server at {base_url}: {e}")
        return

    logger.info(f"Loading transactions from {csv_path} for replay...")
    df = pd.read_csv(csv_path)
    
    if customer_id is not None:
        df = df[df["customer_id"].astype(str) == str(customer_id)]
        logger.info(f"Filtered to customer {customer_id}: {len(df)} transactions available.")
        
    df = df.sort_values("date").head(max_records)
    logger.info(f"Beginning live stream replay of {len(df)} transactions to {api_url} (speed: {delay_sec}s/txn)...\n")
    
    success_count = 0
    alerts_triggered = 0
    recurring_detected_count = 0
    
    print("-" * 80)
    print(f"{'DATE':<12} | {'MERCHANT':<36} | {'AMOUNT':<10} | {'STATUS':<12} | {'FREQUENCY'}")
    print("-" * 80)
    
    for idx, row in df.iterrows():
        payload = {
            "txn_id": str(row["txn_id"]),
            "customer_id": str(row["customer_id"]),
            "date": str(row["date"])[:10],
            "merchant": str(row["merchant"]),
            "amount": float(row["amount"]),
            "txn_type": str(row.get("txn_type", "DEBIT")),
            "category": str(row.get("category", "General")),
            "payment_method": str(row.get("payment_method", "Transfer")),
            "currency": str(row.get("currency", "CZK"))
        }
        
        try:
            resp = requests.post(api_url, json=payload, timeout=5)
            if resp.status_code == 200:
                success_count += 1
                data = resp.json()
                is_rec = data.get("is_recurring", False)
                freq = data.get("frequency", "None")
                status = "RECURRING" if is_rec else "Normal"
                if is_rec:
                    recurring_detected_count += 1
                
                print(f"{payload['date']:<12} | {payload['merchant'][:36]:<36} | {payload['amount']:>7.2f} CZK | {status:<12} | {freq}")
                
                if data.get("alerts"):
                    alerts_triggered += len(data["alerts"])
                    for a in data["alerts"]:
                        print(f"   >>> [ALERT DETECTED]: {a}")
            else:
                logger.error(f"HTTP Error {resp.status_code}: {resp.text}")
        except Exception as e:
            logger.error(f"Failed to stream transaction {row['txn_id']}: {e}")
            break
            
        time.sleep(delay_sec)

    # Injected Events Demonstration
    if inject_events and customer_id is not None:
        print("\n" + "=" * 80)
        print(">>> INJECTING REAL-TIME RISK EVENTS INTO LIVE STREAM <<<")
        print("=" * 80)
        
        # Event 1: Silent Price Hike (> 3% increase on recurring subscription)
        hike_payload = {
            "txn_id": "stream_hike_001",
            "customer_id": str(customer_id),
            "date": "1996-06-05",
            "merchant": "Allianz Insurance (YZ) - Utilities & Household (#87144583)",
            "amount": 2750.00,  # Increased from 2452.00 (+12.2%)
            "category": "Utilities & Household",
            "currency": "CZK"
        }
        resp = requests.post(api_url, json=hike_payload, timeout=5)
        hike_res = resp.json()
        print(f"\n[EVENT 1: Price Hike Injection] - Amount changed from 2,452.00 CZK -> 2,750.00 CZK")
        print(f"HTTP Response: {resp.status_code} | Price Hike Flag: {hike_res.get('is_price_increase')}")
        for a in hike_res.get("alerts", []):
            print(f"   >>> {a}")
            
        # Event 2: Accidental Duplicate Debit (< 24 hours)
        dup_payload = {
            "txn_id": "stream_dup_002",
            "customer_id": str(customer_id),
            "date": "1996-06-05",  # Same date & same amount
            "merchant": "Allianz Insurance (YZ) - Utilities & Household (#87144583)",
            "amount": 2750.00,
            "category": "Utilities & Household",
            "currency": "CZK"
        }
        resp = requests.post(api_url, json=dup_payload, timeout=5)
        dup_res = resp.json()
        print(f"\n[EVENT 2: Duplicate Debit Injection] - Identical 2,750.00 CZK transaction on same date")
        print(f"HTTP Response: {resp.status_code} | Duplicate Flag: {dup_res.get('is_duplicate')}")
        for a in dup_res.get("alerts", []):
            print(f"   >>> {a}")

        # Event 3: Outlier Anomaly
        ano_payload = {
            "txn_id": "stream_ano_003",
            "customer_id": str(customer_id),
            "date": "1996-06-15",
            "merchant": "Luxury Electronic Retailer",
            "amount": 95000.00,  # Extreme outlier amount
            "category": "General Outflow",
            "currency": "CZK"
        }
        resp = requests.post(api_url, json=ano_payload, timeout=5)
        ano_res = resp.json()
        print(f"\n[EVENT 3: Outlier Anomaly Injection] - Unusual 95,000.00 CZK outflow")
        print(f"HTTP Response: {resp.status_code} | Anomaly Flag: {ano_res.get('is_anomaly')} (Score: {ano_res.get('anomaly_score', 0):.4f})")
        for a in ano_res.get("alerts", []):
            print(f"   >>> {a}")

    # Final summary check via GET /customer/{customer_id}/summary
    if customer_id is not None:
        print("\n" + "=" * 80)
        print(f">>> QUERYING API SUMMARY FOR ACCOUNT {customer_id} <<<")
        print("=" * 80)
        summary_resp = requests.get(f"{base_url}/customer/{customer_id}/summary", timeout=5)
        if summary_resp.status_code == 200:
            summary_data = summary_resp.json()
            print(f"Account ID:                 {summary_data.get('customer_id')}")
            print(f"Active Subscriptions:       {summary_data.get('active_subscriptions_count')}")
            print(f"Monthly Subscription Burn:  {summary_data.get('total_monthly_subscription_spend'):,.2f} CZK")
            print(f"Projected Annual Spend:     {summary_data.get('total_yearly_subscription_spend'):,.2f} CZK")
            print(f"Identified Annual Savings:  {summary_data.get('potential_savings'):,.2f} CZK")
            print("Active Subscriptions:")
            for sub in summary_data.get("subscriptions", []):
                print(f"  * {sub['merchant']}: {sub['monthly_cost']:,.2f} CZK/mo [{sub['frequency']}] (Next billing: {sub['next_date']})")
        else:
            logger.error(f"Failed to fetch summary: {summary_resp.text}")

    print("=" * 80)
    logger.info(f"Stream replay completed: {success_count} transactions sent over HTTP. Total alerts captured: {alerts_triggered}")

if __name__ == "__main__":
    replay_stream(max_records=25)
