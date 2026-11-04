# 🛡️ Hidden Subscription & Expense Detector

> An end-to-end Machine Learning and Rule-Engine system designed to automatically uncover hidden recurring subscriptions, silent price increases, accidental duplicate debits, and anomalous outflows from bank transaction statements and real-time payment streams.

---

## 🎬 Live System Demo (Screen Recording)

![Live Application Demo](reports/figures/demo.gif)

> 📹 **Download Full HD Video (720p):** [`reports/figures/demo_video.mp4`](reports/figures/demo_video.mp4)

---

## 📌 Problem Overview

Bank transaction ledgers contain thousands of unformatted rows without an explicit column identifying whether a transaction is a recurring subscription. Consumers lose substantial amounts of money annually due to:
1. **Forgotten or "Zombie" Subscriptions:** Free trials that transitioned to paid billing unnoticed.
2. **Silent Price Hikes:** Incremental rate increases without explicit customer alerts.
3. **Accidental Duplicate Debits:** Glitches resulting in double-charging within hours.
4. **Outlier / Anomalous Debits:** Spikes or unusual outlays that deviate from behavioral norms.

This system provides a production-grade, hybrid pipeline combining explainable rule engines with high-capacity supervised and unsupervised Machine Learning models.

---

## 🏛️ System Architecture

```
[Raw Bank Data / Stream Replay]
               │
               ▼
   [Unified Schema Ingestion]  ───▶ Standardized fields: txn_id, customer_id, date, merchant, amount, category
               │
               ▼
   [Merchant Normalizer]       ───▶ Cleans POS tokens, strips terminal codes & noise
               │
               ▼
   [Feature Engineering]       ───▶ Group metrics: cadence gaps, amount CV, gap CV, DOM stability
               │
               ▼
   [Hybrid Detection Layer]
      ├── Heuristic Rule Engine (Strict regularities & ground truth constraints)
      ├── Supervised Classifier (Random Forest / XGBoost for noisy recurring patterns)
      └── Unsupervised Outlier Detector (Isolation Forest for expense anomalies)
               │
               ▼
   [Inference & Serving Layer]
      ├── FastAPI Microservice (< 10 ms real-time scoring SLA)
      ├── Stream Replay Simulator (Sequential event-driven streaming)
      └── Streamlit Analytics Dashboard (Subscriptions, alerts, and savings optimizer)
```

---

## 📊 Experimental Results

Trained and evaluated on **651,237 debit transactions** using a strict **time-based train/test split** (Train: 1993-01 to 1997-06; Test: 1997-07 to 1998-12) to eliminate lookahead leakage:

| Model / Approach | Precision | Recall | F1-Score | PR-AUC | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Heuristic Rule Baseline** | 1.0000 | 0.5763 | 0.7312 | - | - |
| **Random Forest (Balanced)** | **0.9979** | **0.9997** | **0.9988** | **1.0000** | **1.0000** |

*Target Success Criteria from Blueprint: $F_1 \ge 0.90$. The ML model significantly surpassed the target and recovered over $42\%$ of recurring subscriptions missed by strict rules.*

---

## 🚀 Getting Started

### 1. Installation

Clone the repository and install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Run Data Processing & Model Training

Execute the automated pipeline:
```bash
# 1. Clean and normalize bank transactions
python -m src.data.cleaner

# 2. Train recurring subscription classifier
python -m src.models.train_recurring

# 3. Train unsupervised anomaly detector
python -m src.models.train_anomaly
```

### 3. Launch FastAPI Serving Microservice

Start the high-throughput scoring service:
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive Swagger documentation is available at `http://localhost:8000/docs`.

### 4. Launch Streamlit Analytics Dashboard

Start the interactive dashboard:
```bash
streamlit run app/dashboard.py
```

### 5. Run Test Suite

```bash
pytest tests -v
```

---

## 🔌 API Usage Example

Score a single incoming transaction via `POST /score`:

```python
import requests

payload = {
    "txn_id": "tx_99812",
    "customer_id": "1",
    "date": "1998-05-05",
    "merchant": "Allianz Insurance (YZ)",
    "amount": 2452.00,
    "category": "Insurance"
}

response = requests.post("http://localhost:8000/score", json=payload)
print(response.json())
```

**Response:**
```json
{
  "txn_id": "tx_99812",
  "customer_id": "1",
  "merchant": "Allianz Insurance (YZ)",
  "amount": 2452.0,
  "date": "1998-05-05",
  "is_recurring": true,
  "recurring_probability": 0.998,
  "is_price_increase": false,
  "is_duplicate": false,
  "is_anomaly": false,
  "anomaly_score": 0.34,
  "frequency": "Monthly",
  "alerts": []
}
```

---

## 📁 Repository Structure

```
├── config/
│   └── config.yaml               # Thresholds, hyperparameters & file paths
├── data/
│   ├── raw/                      # Original raw dataset
│   ├── interim/                  # Cleaned transactions in standard schema
│   └── processed/                # Model-ready feature sets
├── src/
│   ├── data/
│   │   ├── loader.py             # Data loader & YAML parser
│   │   ├── cleaner.py            # Transformation & unified schema mapping
│   │   └── schema.py             # Pydantic data schemas
│   ├── features/
│   │   ├── merchant_normalizer.py# Brand normalization & token stripping
│   │   └── build_features.py     # Rolling intervals, CV, and cadence features
│   ├── labeling/
│   │   └── rules.py              # Heuristic rule engines
│   ├── models/
│   │   ├── train_recurring.py    # Supervised recurring payment classifier
│   │   ├── train_anomaly.py      # Unsupervised Isolation Forest model
│   │   └── predict.py            # Real-time stateful inference engine
│   ├── streaming/
│   │   └── replay.py             # Event-driven streaming replay simulation
│   └── api/
│       └── main.py               # FastAPI microservice (< 10ms latency)
├── app/
│   └── dashboard.py              # Streamlit dashboard with Plotly visuals
├── models/
│   ├── recurring_model.joblib    # Trained subscription classifier
│   └── anomaly_model.joblib      # Trained outlier detector
└── tests/
    ├── test_features.py          # Unit tests for feature extraction
    ├── test_rules.py             # Unit tests for rule engine
    └── test_api.py               # Unit tests for API endpoints
```
