import os
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from src.utils.logger import get_logger
from src.data.schema import TransactionInput, ScoreResult, CustomerSummary
from src.labeling.rules import get_cadence_frequency

logger = get_logger("detector_engine")

class ExpenseDetector:
    def __init__(
        self,
        recurring_model_path: str = "models/recurring_model.joblib",
        anomaly_model_path: str = "models/anomaly_model.joblib"
    ):
        self.recurring_model = None
        self.recurring_cols = []
        self.anomaly_model = None
        self.anomaly_cols = []
        self.anomaly_threshold = 0.65
        
        # Load recurring model if present
        if os.path.exists(recurring_model_path):
            artifact = joblib.load(recurring_model_path)
            self.recurring_model = artifact["model"]
            self.recurring_cols = artifact["feature_cols"]
            logger.info("Loaded recurring classifier.")
        else:
            logger.warning(f"Recurring model artifact not found at {recurring_model_path}. Rules will be used exclusively.")

        # Load anomaly model if present
        if os.path.exists(anomaly_model_path):
            artifact = joblib.load(anomaly_model_path)
            self.anomaly_model = artifact["model"]
            self.anomaly_cols = artifact["feature_cols"]
            self.anomaly_threshold = artifact.get("threshold", 0.65)
            logger.info("Loaded anomaly model.")
        
        # In-memory history buffer: (customer_id, merchant) -> list of txn dicts
        self.history: Dict[Tuple[str, str], List[dict]] = {}
        # Alerts buffer: customer_id -> list of alert dicts
        self.alerts_log: Dict[str, List[dict]] = {}

    def _compute_rolling_features(self, history: List[dict], current_txn: dict) -> dict:
        amounts = [t["amount"] for t in history] + [current_txn["amount"]]
        dates = [pd.to_datetime(t["date"]) for t in history] + [pd.to_datetime(current_txn["date"])]
        
        n_txn = len(amounts)
        mean_amt = float(np.mean(amounts))
        std_amt = float(np.std(amounts)) if n_txn > 1 else 0.0
        amt_cv = (std_amt / mean_amt) if mean_amt > 0 else 0.0
        
        if n_txn > 1:
            gaps = [(dates[i] - dates[i - 1]).total_seconds() / (24 * 3600) for i in range(1, n_txn)]
            non_zero_gaps = [g for g in gaps if g >= 1.0]
            mean_gap = float(np.mean(non_zero_gaps)) if non_zero_gaps else float(np.mean(gaps))
            std_gap = float(np.std(non_zero_gaps)) if len(non_zero_gaps) > 1 else 0.0
            gap_cv = (std_gap / mean_gap) if mean_gap > 0 else 1.0
            last_gap = gaps[-1]
            doms = [d.day for d in dates]
            dom_std = float(np.std(doms)) if len(doms) > 1 else 0.0
        else:
            mean_gap = 0.0
            std_gap = 0.0
            gap_cv = 1.0
            last_gap = 0.0
            dom_std = 0.0
            
        amt_to_mean_ratio = (current_txn["amount"] / mean_amt) if mean_amt > 0 else 1.0
        gap_to_mean_ratio = (last_gap / mean_gap) if mean_gap > 0 else 1.0
        
        return {
            "amount": current_txn["amount"],
            "n_txn": n_txn,
            "mean_amt": mean_amt,
            "std_amt": std_amt,
            "amt_cv": amt_cv,
            "mean_gap": mean_gap,
            "std_gap": std_gap,
            "gap_cv": gap_cv,
            "dom_std": dom_std,
            "amt_to_mean_ratio": amt_to_mean_ratio,
            "gap_to_mean_ratio": gap_to_mean_ratio,
            "gap_days": last_gap,
            "is_approx_monthly": 1 if 26 <= mean_gap <= 35 else 0,
            "is_approx_weekly": 1 if 6 <= mean_gap <= 8 else 0,
            "is_approx_yearly": 1 if 355 <= mean_gap <= 375 else 0
        }

    def score_transaction(self, t: TransactionInput) -> ScoreResult:
        key = (str(t.customer_id), str(t.merchant))
        past_txns = self.history.get(key, [])
        t_dict = t.model_dump() if hasattr(t, "model_dump") else t.dict()
        current_date = pd.to_datetime(t.date)
        
        feats = self._compute_rolling_features(past_txns, t_dict)
        alerts: List[str] = []
        
        # 1. Duplicate Detection Check
        is_duplicate = False
        for past in reversed(past_txns):
            past_date = pd.to_datetime(past["date"])
            diff_hours = (current_date - past_date).total_seconds() / 3600.0
            if 0 <= diff_hours <= 24.0 and abs(past["amount"] - t.amount) < 0.01:
                is_duplicate = True
                alerts.append(f"POSSIBLE DUPLICATE: Same charge of {t.amount} {t.currency} to {t.merchant} within {diff_hours:.1f} hours.")
                break
        
        # 2. Recurring Detection (Rules + ML)
        rule_recurring = (
            feats["n_txn"] >= 3 and
            (feats["is_approx_monthly"] == 1 or feats["is_approx_weekly"] == 1 or feats["is_approx_yearly"] == 1) and
            feats["gap_cv"] <= 0.25 and
            feats["amt_cv"] <= 0.20
        )
        
        rec_prob = 0.0
        if self.recurring_model is not None:
            input_df = pd.DataFrame([{c: feats.get(c, 0) for c in self.recurring_cols}])
            rec_prob = float(self.recurring_model.predict_proba(input_df)[0][1])
            is_recurring = (rec_prob >= 0.50) or rule_recurring
        else:
            is_recurring = rule_recurring
            rec_prob = 1.0 if rule_recurring else 0.0
            
        freq_label = get_cadence_frequency(feats["mean_gap"]) if is_recurring else "None"
        
        # 3. Price Increase Detection
        is_price_increase = False
        if is_recurring and len(past_txns) >= 2:
            prev_amt = past_txns[-1]["amount"]
            if prev_amt > 0:
                pct_hike = (t.amount - prev_amt) / prev_amt
                if pct_hike >= 0.03:
                    is_price_increase = True
                    alerts.append(f"PRICE INCREASE: Subscription amount grew {pct_hike:.1%} (from {prev_amt:.2f} to {t.amount:.2f} {t.currency}).")
        
        # 4. Anomaly Detection
        is_anomaly = False
        anomaly_score = 0.0
        if self.anomaly_model is not None:
            ano_df = pd.DataFrame([{c: feats.get(c, 0) for c in self.anomaly_cols}])
            raw_score = -float(self.anomaly_model.score_samples(ano_df)[0])
            anomaly_score = raw_score
            if raw_score >= self.anomaly_threshold or feats["amt_to_mean_ratio"] >= 3.0:
                is_anomaly = True
                alerts.append(f"UNUSUAL TRANSACTION: Outflow of {t.amount:.2f} {t.currency} is unusually high compared to typical history.")
        else:
            if feats["amt_to_mean_ratio"] >= 3.0:
                is_anomaly = True
                anomaly_score = feats["amt_to_mean_ratio"]
                alerts.append(f"UNUSUAL TRANSACTION: Outflow is 3x higher than historical average.")

        # Update customer state history
        self.history.setdefault(key, []).append(t_dict)
        if alerts:
            self.alerts_log.setdefault(str(t.customer_id), []).extend([
                {"date": t.date, "merchant": t.merchant, "amount": t.amount, "message": a}
                for a in alerts
            ])
            
        return ScoreResult(
            txn_id=t.txn_id,
            customer_id=t.customer_id,
            merchant=t.merchant,
            amount=t.amount,
            date=t.date,
            is_recurring=is_recurring,
            recurring_probability=rec_prob,
            is_price_increase=is_price_increase,
            is_duplicate=is_duplicate,
            is_anomaly=is_anomaly,
            anomaly_score=anomaly_score,
            frequency=freq_label,
            alerts=alerts
        )

    def get_customer_summary(self, customer_id: str) -> CustomerSummary:
        cust_str = str(customer_id)
        cust_keys = [k for k in self.history.keys() if k[0] == cust_str]
        
        subscriptions = []
        monthly_total = 0.0
        
        for key in cust_keys:
            txns = self.history[key]
            merchant = key[1]
            last_txn = txns[-1]
            feats = self._compute_rolling_features(txns[:-1], last_txn)
            
            rule_rec = (feats["n_txn"] >= 3 and feats["is_approx_monthly"] == 1 and feats["amt_cv"] <= 0.20)
            if self.recurring_model is not None:
                input_df = pd.DataFrame([{c: feats.get(c, 0) for c in self.recurring_cols}])
                is_rec = bool(self.recurring_model.predict(input_df)[0]) or rule_rec
            else:
                is_rec = rule_rec
                
            if is_rec:
                freq = get_cadence_frequency(feats["mean_gap"])
                amt = last_txn["amount"]
                # Convert to monthly equivalent
                if freq == "Weekly":
                    m_cost = amt * 4.33
                elif freq == "Yearly":
                    m_cost = amt / 12.0
                else:
                    m_cost = amt
                    
                monthly_total += m_cost
                
                # Estimate next billing date
                last_dt = pd.to_datetime(last_txn["date"])
                gap = int(round(feats["mean_gap"])) if feats["mean_gap"] > 0 else 30
                next_date = (last_dt + timedelta(days=gap)).strftime("%Y-%m-%d")
                
                subscriptions.append({
                    "merchant": merchant,
                    "amount": amt,
                    "frequency": freq,
                    "monthly_cost": round(m_cost, 2),
                    "last_date": last_txn["date"],
                    "next_date": next_date,
                    "n_payments": feats["n_txn"]
                })
        
        recent_alerts = [a["message"] for a in self.alerts_log.get(cust_str, [])][-5:]
        yearly_total = monthly_total * 12.0
        # Potential savings: top 20% most expensive subscriptions
        potential_savings = sum(s["monthly_cost"] for s in sorted(subscriptions, key=lambda x: x["monthly_cost"], reverse=True)[:1]) * 12.0 if subscriptions else 0.0
        
        return CustomerSummary(
            customer_id=cust_str,
            total_monthly_subscription_spend=round(monthly_total, 2),
            total_yearly_subscription_spend=round(yearly_total, 2),
            active_subscriptions_count=len(subscriptions),
            subscriptions=subscriptions,
            recent_alerts=recent_alerts,
            potential_savings=round(potential_savings, 2)
        )
