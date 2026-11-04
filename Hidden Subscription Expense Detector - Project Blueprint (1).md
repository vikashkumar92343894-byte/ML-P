# Hidden Subscription Expense Detector

Project blueprint: problem statement, architecture, implementation, folder structure

## 1. Problem statement

People lose money to subscriptions they forgot about, silent price hikes, double charges and unusual transactions. A bank statement has thousands of rows and no column that says this is a subscription.

**Goal:** build an ML-based system that reads bank transactions (as a file or as a live stream) and automatically:

1. Detects recurring payments and subscriptions
2. Flags price increases on those subscriptions
3. Flags possible duplicate charges
4. Flags anomalous transactions
5. Reports total monthly and yearly subscription cost, with savings suggestions

**Success criteria**

- Recurring detection: F1 of 0.90 or higher on a held-out test set
- Anomaly detection: recall is the priority, with PR-AUC reported
- Scoring one transaction through the API takes under 200 ms

## 2. Datasets and their roles

| Dataset | Role in the project |
| --- | --- |
| Hidden\_Subscription\_Expense\_Detector\_Dataset.csv (1,469 rows, INR, 4 label columns) | Main training and testing data. It has no customer column, so treat it as one account and group by Merchant |
| Bank\_Transactions (20,000 rows, has CustomerID and IsFraud) | Build rule-based labels and test how well the model generalizes. Do not mix with the INR file without converting currency |
| Berka (Czech bank, trans and order tables) | Validate your recurring logic on real data. It has no merchant names |
| Your own anonymized UPI or bank statements | Final real-world test |

**Leakage warning:** in your labeled CSV, columns such as Description and Anomaly\_Reason may reveal the answer. Never use Anomaly\_Reason as a feature, and check whether Description gives away the label.

## 3. Architecture

```
[Data sources]
 CSV files | Berka | own statements | replay stream
        |
        v
[Ingestion + cleaning]  -> unified schema, date parsing, currency column
        |
        v
[Merchant normalization] -> NETFLIX*1234 becomes netflix
        |
        v
[Feature engineering]   -> gaps, amount stability, price change, duplicates
        |
        v
[Detection layer]
  Rule engine  +  ML models (recurring, anomaly)
        |
        v
[Scoring API - FastAPI]  <---  [Replay streamer: sends 1 txn at a time]
        |
        v
[Dashboard - Streamlit]  -> subscriptions list, alerts, monthly cost, savings
```

**Why a hybrid of rules and ML:** rules give explainable labels (for example, a roughly 30-day gap with a stable amount). ML learns the messy cases that rules miss. Interviewers like this because it mirrors how production systems work.

## 4. Tech stack

- Language: Python 3.10+
- Data: Pandas, NumPy
- ML: scikit-learn, XGBoost or LightGBM, imbalanced-learn
- Explainability: SHAP
- Serving: FastAPI, Uvicorn
- Dashboard: Streamlit, Plotly
- Quality: pytest, black, Git and GitHub
- Optional: Docker, Kafka (only if you want a true streaming demo)

## 5. Folder structure

```
hidden-subscription-detector/
|-- README.md
|-- requirements.txt
|-- .gitignore
|-- Dockerfile
|-- config/
|   |-- config.yaml            # paths, thresholds, column mappings
|-- data/
|   |-- raw/                   # original files, never edited
|   |-- interim/               # cleaned data
|   |-- processed/             # feature tables, model-ready
|   |-- external/              # Berka tables
|-- notebooks/
|   |-- 01_eda.ipynb
|   |-- 02_feature_engineering.ipynb
|   |-- 03_modeling.ipynb
|   |-- 04_evaluation.ipynb
|-- src/
|   |-- __init__.py
|   |-- data/
|   |   |-- loader.py
|   |   |-- cleaner.py
|   |   |-- schema.py
|   |-- features/
|   |   |-- merchant_normalizer.py
|   |   |-- build_features.py
|   |-- labeling/
|   |   |-- rules.py           # rule-based labels
|   |-- models/
|   |   |-- train_recurring.py
|   |   |-- train_anomaly.py
|   |   |-- predict.py
|   |-- streaming/
|   |   |-- replay.py          # simulates real-time feed
|   |-- api/
|   |   |-- main.py            # FastAPI app
|   |-- utils/
|       |-- logger.py
|-- app/
|   |-- dashboard.py           # Streamlit
|-- models/                    # saved .joblib files
|-- reports/
|   |-- figures/
|-- tests/
|   |-- test_features.py
|   |-- test_rules.py
|   |-- test_api.py
|-- docs/
    |-- architecture.png
```

## 6. Step-by-step implementation

### Phase 0: Setup

1. Create the GitHub repo and the folders above.
2. Create a virtual environment and install: pandas, numpy, scikit-learn, xgboost, imbalanced-learn, shap, fastapi, uvicorn, streamlit, plotly, pytest, pyyaml, joblib.
3. Put raw files in data/raw/ and never modify them.

### Phase 1: Exploratory data analysis (notebook 01)

- Shape, dtypes, missing values, date range
- Class balance for Is\_Recurring\_Payment, Is\_Price\_Increase, Is\_Possible\_Duplicate and Is\_Anomaly. These are likely very imbalanced, which decides your metrics
- Top merchants, spend per category, transactions per month
- Plot the day gaps between payments for a few recurring merchants

### Phase 2: Cleaning and a unified schema

Convert every dataset to the same column names so one pipeline works for all of them:

`txn_id, customer_id, date, merchant, amount, txn_type, category, payment_method, currency`

- Parse dates to datetime and sort by date
- Keep only Debit rows for subscription detection
- Drop failed transactions from recurring detection
- For the labeled CSV, set customer\_id to a constant because it is one account
- Add a currency column (INR or USD) and never mix currencies

### Phase 3: Merchant normalization

Bank descriptions are messy (NETFLIX.COM 1234, Netflix India). Lowercase, strip digits and symbols, remove words like pvt, ltd and upi, then fuzzy-match to a known merchant list (rapidfuzz works well). Without this, one subscription gets split into many groups.

### Phase 4: Feature engineering

The unit of analysis is a customer plus merchant group.

```python
import pandas as pd

def build_group_features(df, keys=('customer_id', 'merchant')):
    keys = list(keys)
    df = df.sort_values(keys + ['date']).copy()
    df['gap_days'] = df.groupby(keys)['date'].diff().dt.days
    feats = df.groupby(keys).agg(
        n_txn=('amount', 'size'),
        mean_amt=('amount', 'mean'),
        std_amt=('amount', 'std'),
        mean_gap=('gap_days', 'mean'),
        std_gap=('gap_days', 'std'),
        first_date=('date', 'min'),
        last_date=('date', 'max'),
    ).reset_index()
    feats['amt_cv'] = (feats['std_amt'] / feats['mean_amt']).fillna(0)
    feats['gap_cv'] = (feats['std_gap'] / feats['mean_gap']).fillna(1)
    return feats
```

Other useful features: day-of-month consistency, number of distinct amounts, share of payments within 3 days of the expected date, last amount versus first amount, category, payment method, and amount z-score within its category.

### Phase 5: Rule-based labeling (for datasets with no labels)

```python
def label_recurring(feats, min_txn=3):
    monthly = feats['mean_gap'].between(26, 35)
    weekly = feats['mean_gap'].between(6, 8)
    yearly = feats['mean_gap'].between(355, 375)
    regular = (feats['gap_cv'] < 0.2) & (feats['amt_cv'] < 0.25)
    return (feats['n_txn'] >= min_txn) & (monthly | weekly | yearly) & regular
```

- **Price increase:** inside a recurring group, the amount rises by more than 3 percent versus the previous payment and the new amount then stays stable.
- **Duplicate:** same customer, merchant and amount appearing again within 24 hours.
- **Anomaly:** amount far from the merchant or category norm, or a first-time merchant with a large amount.

```python
# price increase
g = df.groupby(['customer_id', 'merchant'])
df['prev_amt'] = g['amount'].shift(1)
df['pct_change'] = (df['amount'] - df['prev_amt']) / df['prev_amt']
df['price_up'] = df['is_recurring'] & (df['pct_change'] > 0.03)

# duplicate
k = ['customer_id', 'merchant', 'amount']
df['hrs_since_same'] = df.groupby(k)['date'].diff().dt.total_seconds() / 3600
df['dup'] = df['hrs_since_same'] < 24
```

### Phase 6: Modeling (notebook 03)

| Task | Approach |
| --- | --- |
| Recurring | Random Forest or XGBoost on group features. Baseline: the rule above |
| Price increase | Rule plus a classifier on pct\_change, history length and category |
| Duplicate | Rule-first. A classifier is optional given how rare duplicates are |
| Anomaly | Isolation Forest (unsupervised) and a supervised model using Is\_Anomaly |

Important practices:

- Use a **time-based split** (train on earlier months, test on later ones) so the model never sees the future
- For imbalance, use class\_weight='balanced' first, then SMOTE if needed
- Always compare against the simple rule baseline. If ML does not beat it, say so honestly

```python
from sklearn.ensemble import IsolationForest

iso = IsolationForest(contamination=0.02, random_state=42)
df['anomaly_score'] = -iso.fit(X_train).score_samples(X_all)
```

### Phase 7: Evaluation (notebook 04)

- Report precision, recall, F1 and PR-AUC per task. Accuracy is misleading on imbalanced data
- Show the confusion matrix and the errors you could not fix
- Use SHAP to explain which features drive each prediction
- Run the final model on Berka and on your own statements to show generalization

### Phase 8: Real-time simulation

**Scoring API (src/api/main.py)**

```python
from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()
model = joblib.load('models/recurring_model.joblib')
history = {}  # (customer, merchant) -> list of past txns

class Txn(BaseModel):
    txn_id: str
    customer_id: str
    date: str
    merchant: str
    amount: float

@app.post('/score')
def score(t: Txn):
    key = (t.customer_id, t.merchant)
    history.setdefault(key, []).append(t.dict())
    # rebuild group features from history, then model.predict_proba(...)
    return {'txn_id': t.txn_id, 'is_recurring': False, 'alerts': []}
```

**Replay streamer (src/streaming/replay.py)**

```python
import json, time, requests
import pandas as pd

df = pd.read_csv('data/processed/transactions.csv').sort_values('date')
for _, row in df.iterrows():
    payload = json.loads(row.to_json())
    requests.post('http://localhost:8000/score', json=payload, timeout=5)
    time.sleep(0.2)
```

This feeds transactions one by one, exactly like a live bank feed. For an advanced version, put Kafka between the streamer and the API.

### Phase 9: Dashboard (app/dashboard.py)

Build with Streamlit and show:

- Detected subscriptions table (merchant, amount, frequency, next expected date)
- Total monthly and yearly subscription cost
- Alerts feed: price increases, duplicates, anomalies
- Spend-by-category chart and a timeline of payments
- Potential savings: list the least-used or most expensive subscriptions

### Phase 10: Testing, packaging and deployment

- pytest for rules and feature functions (use tiny hand-made examples)
- Dockerfile for the API and dashboard
- Deploy the dashboard on Streamlit Community Cloud and keep the code on GitHub

## 7. Four-week timeline

| Week | Work |
| --- | --- |
| 1 | Setup, EDA, cleaning, unified schema, merchant normalization |
| 2 | Features, rule-based labels, baseline models |
| 3 | Final models, evaluation, SHAP, test on Berka and own data |
| 4 | API, replay streamer, dashboard, README, deployment |

## 8. Common pitfalls

- **Data leakage:** features built from label columns, or random splits instead of time-based ones
- **Mixing currencies** (INR and USD) in one model
- **Trusting accuracy** when 95 percent or more of rows are normal
- **Skipping merchant normalization,** which breaks grouping
- **Treating monthly groceries as subscriptions:** require a stable amount, not only a regular gap
- **Training on all data and testing on the same data**

## 9. README and resume tips

**README should include:** problem statement, architecture diagram, dataset sources, how to run (three commands), results table, screenshots of the dashboard, and limitations.

**Sample resume bullet:**

Built a hybrid rule and ML system in Python that detects recurring subscriptions, price hikes, duplicate charges and anomalies from bank transactions, with a FastAPI scoring service, a real-time replay simulator and a Streamlit dashboard (F1 of X on held-out data).

Replace X with your real result. Never quote a number you did not measure.
