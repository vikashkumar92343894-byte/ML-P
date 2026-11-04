import json
import os

os.makedirs("notebooks", exist_ok=True)

def make_nb(cells):
    return {
        "cells": cells,
        "metadata": {
            "language_info": {"name": "python", "version": "3.11"}
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }

def md_cell(text):
    return {"cell_type": "markdown", "metadata": {}, "source": [text]}

def code_cell(code):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [code]}

# 01_eda.ipynb
nb1 = make_nb([
    md_cell("# 01. Exploratory Data Analysis (EDA)\n## Hidden Subscription & Outlier Expense Detector\nThis notebook analyzes transaction distributions, payment frequencies, and cadence patterns across accounts."),
    code_cell("""import pandas as pd
import numpy as np

# Load transactions and standing orders
trans = pd.read_csv('../data/raw/trans.csv', sep=';', low_memory=False)
order = pd.read_csv('../data/raw/order.csv', sep=';', low_memory=False)

print(f'Transactions: {trans.shape}, Standing Orders: {order.shape}')
trans.head()"""),
    md_cell("### 1. Breakdown by Outflow Operation and Categories"),
    code_cell("""print('Transaction Types:')
print(trans['type'].value_counts())

print('\\nOperations:')
print(trans['operation'].value_counts())

print('\\nCategories (k_symbol):')
print(trans['k_symbol'].value_counts(dropna=False))"""),
    md_cell("### 2. Payment Intervals (Gaps) for Bank Remittances"),
    code_cell("""trans['date_dt'] = pd.to_datetime(trans['date'].astype(str), format='%y%m%d')
remit = trans[trans['operation'] == 'PREVOD NA UCET'].sort_values(['account_id', 'bank', 'account', 'date_dt'])
remit['gap'] = remit.groupby(['account_id', 'bank', 'account'])['date_dt'].diff().dt.days

print('Inter-payment Gap Summary:')
print(remit['gap'].describe())
print('\\nTop Gaps:')
print(remit['gap'].value_counts().head(10))""")
])

with open("notebooks/01_eda.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb1, f, indent=1)

# 02_feature_engineering.ipynb
nb2 = make_nb([
    md_cell("# 02. Feature Engineering & Merchant Normalization\nBuilding rolling features, gap CV, and amount stability metrics."),
    code_cell("""import sys
sys.path.append('..')
import pandas as pd
from src.features.merchant_normalizer import normalize_merchant
from src.features.build_features import build_features

df = pd.read_csv('../data/interim/clean_transactions.csv', nrows=25000)
featured = build_features(df)
print(f'Computed {featured.shape[1]} features across {len(featured)} records.')
featured[['amount', 'mean_amt', 'amt_cv', 'mean_gap', 'gap_cv', 'is_approx_monthly']].head()""")
])

with open("notebooks/02_feature_engineering.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb2, f, indent=1)

# 03_modeling.ipynb
nb3 = make_nb([
    md_cell("# 03. Machine Learning Modeling\nTraining Random Forest and Isolation Forest models on time-based split."),
    code_cell("""import sys
sys.path.append('..')
from src.models.train_recurring import train_recurring_model
from src.models.train_anomaly import train_anomaly_model

rec_results = train_recurring_model(
    data_path='../data/interim/clean_transactions.csv',
    model_output_path='../models/recurring_model.joblib'
)
print('Recurring Classifier Results:', rec_results)

ano_results = train_anomaly_model(
    data_path='../data/interim/clean_transactions.csv',
    model_output_path='../models/anomaly_model.joblib'
)
print('Anomaly Model Results:', ano_results)""")
])

with open("notebooks/03_modeling.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb3, f, indent=1)

# 04_evaluation.ipynb
nb4 = make_nb([
    md_cell("# 04. Evaluation & Stateful Inference\nEvaluating latency, alerts, and customer summaries."),
    code_cell("""import sys
sys.path.append('..')
from src.models.predict import ExpenseDetector
from src.data.schema import TransactionInput

detector = ExpenseDetector('../models/recurring_model.joblib', '../models/anomaly_model.joblib')

test_t = TransactionInput(
    txn_id='eval_001',
    customer_id='1',
    date='1997-08-05',
    merchant='Allianz Insurance (YZ)',
    amount=2452.0
)
res = detector.score_transaction(test_t)
print('Scored Result:', res.model_dump())

summary = detector.get_customer_summary('1')
print('Customer Summary:', summary.model_dump())""")
])

with open("notebooks/04_evaluation.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb4, f, indent=1)

print("All 4 notebooks successfully created.")
