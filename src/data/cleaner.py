import os
import pandas as pd
import numpy as np
from src.utils.logger import get_logger
from src.data.loader import load_raw_data, load_config

logger = get_logger("data_cleaner")

# Mapping Czech bank codes to realistic merchant brand names
BANK_MERCHANT_MAP = {
    "YZ": "Allianz Insurance (YZ)",
    "QR": "Ceska Pojistovna (QR)",
    "AB": "Prague Power & Electric (AB)",
    "ST": "O2 Telecom Services (ST)",
    "UV": "Prazska Plynarenska Gas (UV)",
    "WX": "Generali Ceska (WX)",
    "KL": "T-Mobile Czech (KL)",
    "IJ": "Vodafone Czechia (IJ)",
    "GH": "Ceska Posta SIPO (GH)",
    "OP": "Prague District Heating (OP)",
    "EF": "Uniqa Insurance (EF)",
    "MN": "CSOB Leasing & Auto (MN)",
    "CD": "KB Hypotecni Banka (CD)"
}

CATEGORY_MAP = {
    "POJISTNE": "Insurance",
    "SIPO": "Utilities & Household",
    "UVER": "Loan Repayment",
    "LEASING": "Leasing",
    "SLUZBY": "Bank & Service Fees",
    "SANKC. UROK": "Penalties & Fees",
    "DUCHOD": "Pension",
    "UROK": "Interest"
}

def parse_berka_date(date_series: pd.Series) -> pd.Series:
    """Converts YYMMDD integers into pd.Timestamp (1993-1998 range)."""
    return pd.to_datetime(date_series.astype(str), format="%y%m%d")

def clean_berka_transactions(trans: pd.DataFrame, order: pd.DataFrame, sample_size: int = None) -> pd.DataFrame:
    logger.info("Starting cleaning and schema normalization...")
    
    # 1. Focus on debit / expense transactions (VYDAJ and VYBER)
    df = trans[trans["type"].isin(["VYDAJ", "VYBER"])].copy()
    logger.info(f"Filtered to debits/outflows: {len(df):,} transactions")
    
    # 2. Parse dates
    df["date"] = parse_berka_date(df["date"])
    df = df.sort_values(["account_id", "date"]).reset_index(drop=True)
    
    # 3. Clean and map categories
    df["k_symbol"] = df["k_symbol"].fillna("").astype(str).str.strip()
    df["category"] = df["k_symbol"].map(CATEGORY_MAP).fillna("General Outflow")
    
    # 4. Synthesize merchant names
    df["bank"] = df["bank"].fillna("").astype(str).str.strip()
    df["account"] = df["account"].fillna(0).astype(np.int64).astype(str)
    
    def assign_merchant(row):
        op = str(row["operation"])
        k_sym = str(row["k_symbol"])
        bank = str(row["bank"])
        acc = str(row["account"])
        
        if op == "PREVOD NA UCET":
            # Recurring bank transfer remittance
            brand = BANK_MERCHANT_MAP.get(bank, f"Partner Bank {bank}")
            category_tag = CATEGORY_MAP.get(k_sym, "Transfer")
            return f"{brand} - {category_tag} (#{acc})"
        elif op == "VYBER KARTOU":
            return f"Card POS Terminal - Acc {row['account_id']}"
        elif op == "VYBER":
            return "ATM Cash Withdrawal"
        elif k_sym == "SLUZBY":
            return "Bank Account Maintenance Fee"
        else:
            return f"Counterparty Transfer {bank}_{acc}" if bank else "General Direct Debit"

    df["merchant"] = df.apply(assign_merchant, axis=1)
    
    # 5. Map payment method
    def assign_payment_method(op):
        if op == "PREVOD NA UCET":
            return "Standing Order / Transfer"
        elif op == "VYBER KARTOU":
            return "Debit Card"
        elif op == "VYBER":
            return "ATM Cash"
        else:
            return "Direct Debit"

    df["payment_method"] = df["operation"].apply(assign_payment_method)
    
    # 6. Flag Ground Truth Recurring Subscriptions from order.csv
    # Build set of (account_id, bank_to, account_to) from order.csv
    order_clean = order.copy()
    order_clean["bank_to"] = order_clean["bank_to"].fillna("").astype(str).str.strip()
    order_clean["account_to"] = order_clean["account_to"].fillna(0).astype(np.int64).astype(str)
    
    standing_order_keys = set(
        zip(order_clean["account_id"], order_clean["bank_to"], order_clean["account_to"])
    )
    
    trans_keys = list(zip(df["account_id"], df["bank"], df["account"]))
    df["is_ground_truth_recurring"] = [
        1 if (k in standing_order_keys or row_cat == "Bank & Service Fees") else 0
        for k, row_cat in zip(trans_keys, df["category"])
    ]
    
    # Standardize column naming according to blueprint
    df = df.rename(columns={
        "trans_id": "txn_id",
        "account_id": "customer_id"
    })
    
    df["currency"] = "CZK"
    df["txn_type"] = "DEBIT"
    
    selected_cols = [
        "txn_id", "customer_id", "date", "merchant", "amount", 
        "txn_type", "category", "payment_method", "currency", 
        "is_ground_truth_recurring"
    ]
    df = df[selected_cols]
    
    if sample_size and len(df) > sample_size:
        logger.info(f"Subsampling {sample_size:,} rows for rapid development...")
        # Stratify by customer so history chains remain intact
        customers = df["customer_id"].drop_duplicates()
        n_cust = int(len(customers) * (sample_size / len(df)))
        sampled_cust = set(customers.iloc[:n_cust])
        df = df[df["customer_id"].isin(sampled_cust)].copy()
    
    logger.info(f"Cleaned dataset ready: {len(df):,} rows. Ground truth recurring: {df['is_ground_truth_recurring'].sum():,} ({df['is_ground_truth_recurring'].mean():.2%})")
    return df

def run_clean_pipeline(config_path: str = "config/config.yaml", output_path: str = "data/interim/clean_transactions.csv"):
    config = load_config(config_path)
    trans, order, account = load_raw_data(config)
    clean_df = clean_berka_transactions(trans, order)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    logger.info(f"Saving cleaned transactions to {output_path}...")
    clean_df.to_csv(output_path, index=False)
    logger.info("Data cleaning completed successfully.")
    return clean_df

if __name__ == "__main__":
    run_clean_pipeline()
