import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import sys

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.predict import ExpenseDetector
from src.data.schema import TransactionInput

st.set_page_config(
    page_title="Hidden Subscription & Expense Detector",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #1E293B;
        border-radius: 10px;
        padding: 16px;
        color: white;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
    }
    .alert-box {
        padding: 12px;
        border-radius: 8px;
        margin-bottom: 8px;
        border-left: 5px solid #EF4444;
        background-color: #FEF2F2;
        color: #991B1B;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_detector():
    return ExpenseDetector()

@st.cache_data
def load_sample_data():
    clean_path = "data/interim/clean_transactions.csv"
    if os.path.exists(clean_path):
        df = pd.read_csv(clean_path)
        df["date"] = pd.to_datetime(df["date"])
        return df
    return None

detector = get_detector()
df = load_sample_data()

# Sidebar
st.sidebar.title("💳 Expense Detector")
st.sidebar.caption("AI-Powered Subscription & Outflow Intelligence")

if df is not None:
    available_customers = df["customer_id"].unique()[:50]
    selected_customer = st.sidebar.selectbox("Select Account / Customer ID", available_customers)
    
    # Filter dataset for customer
    cust_df = df[df["customer_id"] == selected_customer].sort_values("date")
else:
    cust_df = pd.DataFrame()
    selected_customer = "1"

st.sidebar.markdown("---")
st.sidebar.subheader("Quick Actions")
if st.sidebar.button("Re-process Customer Ledger"):
    detector.history.clear()
    detector.alerts_log.clear()
    for _, row in cust_df.iterrows():
        t = TransactionInput(
            txn_id=str(row["txn_id"]),
            customer_id=str(row["customer_id"]),
            date=str(row["date"])[:10],
            merchant=str(row["merchant"]),
            amount=float(row["amount"]),
            txn_type=str(row["txn_type"]),
            category=str(row["category"]),
            payment_method=str(row["payment_method"]),
            currency="CZK"
        )
        detector.score_transaction(t)
    st.sidebar.success("Ledger parsed and memory updated!")

# Main Title
st.title("🛡️ Hidden Subscription & Outlier Detector")
st.markdown("Automated detection of recurring commitments, silent price hikes, double debits, and anomalous outlays.")

# If memory empty for customer, preload transactions
cust_key_exists = any(k[0] == str(selected_customer) for k in detector.history.keys())
if not cust_key_exists and not cust_df.empty:
    for _, row in cust_df.iterrows():
        t = TransactionInput(
            txn_id=str(row["txn_id"]),
            customer_id=str(row["customer_id"]),
            date=str(row["date"])[:10],
            merchant=str(row["merchant"]),
            amount=float(row["amount"]),
            txn_type=str(row["txn_type"]),
            category=str(row["category"]),
            payment_method=str(row["payment_method"]),
            currency="CZK"
        )
        detector.score_transaction(t)

summary = detector.get_customer_summary(selected_customer)

# Top KPI Metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Active Subscriptions", f"{summary.active_subscriptions_count}", help="Detected monthly recurring services")
with col2:
    st.metric("Monthly Recurring Spend", f"{summary.total_monthly_subscription_spend:,.2f} CZK", delta=None)
with col3:
    st.metric("Projected Annual Spend", f"{summary.total_yearly_subscription_spend:,.2f} CZK")
with col4:
    st.metric("Identified Savings", f"{summary.potential_savings:,.2f} CZK/yr", delta="Optimization")

st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs([
    "📋 Detected Subscriptions", 
    "🚨 Risk & Alerts Center", 
    "📊 Spending Analytics", 
    "🧪 Live Transaction Tester"
])

with tab1:
    st.subheader("Identified Recurring Commitments")
    if summary.subscriptions:
        sub_df = pd.DataFrame(summary.subscriptions)
        sub_df = sub_df.rename(columns={
            "merchant": "Merchant / Service",
            "amount": "Last Charge (CZK)",
            "monthly_cost": "Normalized Monthly (CZK)",
            "frequency": "Cadence",
            "last_date": "Last Date",
            "next_date": "Next Expected Date",
            "n_payments": "Payments Tracked"
        })
        st.dataframe(sub_df, use_container_width=True)
    else:
        st.info("No recurring commitments detected for this account yet.")

with tab2:
    st.subheader("Actionable Alerts Feed")
    alerts = detector.alerts_log.get(str(selected_customer), [])
    if alerts:
        for alert in alerts[-10:]:
            st.warning(f"**[{alert['date']}] {alert['merchant']}** ({alert['amount']:.2f} CZK): {alert['message']}")
    else:
        st.success("🎉 No suspicious anomalies, duplicate charges, or silent price hikes detected.")

with tab3:
    st.subheader("Financial Outflow Intelligence")
    if not cust_df.empty:
        c1, c2 = st.columns([2, 1])
        with c1:
            st.markdown("##### Outflow Timeline by Category")
            fig_time = px.scatter(
                cust_df,
                x="date",
                y="amount",
                color="category",
                hover_data=["merchant", "payment_method"],
                title="Historical Outflows",
                labels={"amount": "Amount (CZK)", "date": "Date"}
            )
            fig_time.update_traces(marker=dict(size=8, opacity=0.8))
            st.plotly_chart(fig_time, use_container_width=True)
        with c2:
            st.markdown("##### Spend Breakdown by Category")
            cat_sum = cust_df.groupby("category")["amount"].sum().reset_index()
            fig_pie = px.pie(
                cat_sum,
                names="category",
                values="amount",
                hole=0.45,
                title="Total Spend Distribution"
            )
            st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("No transaction data available for analysis.")

with tab4:
    st.subheader("Simulate Real-time Transaction Scoring")
    st.markdown("Submit a mock or new transaction to test the hybrid rules and ML model response in real time.")
    
    with st.form("simulate_txn_form"):
        sc1, sc2 = st.columns(2)
        with sc1:
            sim_merchant = st.selectbox(
                "Merchant / Counterparty",
                ["Allianz Insurance (YZ)", "Prague Power & Electric (AB)", "O2 Telecom Services (ST)", "Ceska Posta SIPO (GH)", "Netflix Video", "Spotify Music", "Custom Merchant"]
            )
            if sim_merchant == "Custom Merchant":
                sim_merchant = st.text_input("Enter Merchant Name", "Amazon Prime")
            sim_amt = st.number_input("Amount (CZK)", min_value=1.0, value=2452.0, step=10.0)
        with sc2:
            sim_date = st.date_input("Transaction Date")
            sim_cat = st.selectbox("Category", ["Insurance", "Utilities & Household", "Entertainment", "Bank & Service Fees", "General Outflow"])
        
        submitted = st.form_submit_button("⚡ Score Transaction")
        if submitted:
            test_t = TransactionInput(
                txn_id=f"sim_{np.random.randint(100000, 999999)}",
                customer_id=str(selected_customer),
                date=sim_date.strftime("%Y-%m-%d"),
                merchant=sim_merchant,
                amount=float(sim_amt),
                category=sim_cat
            )
            res = detector.score_transaction(test_t)
            
            st.markdown("#### Model Scoring Output")
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Recurring Flag", "YES" if res.is_recurring else "NO")
            mc2.metric("ML Recurring Probability", f"{res.recurring_probability:.1%}")
            mc3.metric("Anomaly Flag", "ANOMALY" if res.is_anomaly else "NORMAL")
            
            if res.alerts:
                st.error("🚨 Triggered Alerts: " + " | ".join(res.alerts))
            else:
                st.success("✅ Transaction is clean and within expected behavioral boundaries.")
