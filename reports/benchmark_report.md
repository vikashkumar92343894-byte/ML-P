# 📈 Experimental Results & Benchmark Report

## 1. Overview & Setup
* **Dataset:** Berka Czech Financial Dataset (Debit Outflows)
* **Total Transactions Evaluated:** 291,980 held-out records
* **Validation Strategy:** Temporal / Time-Based Split (`date >= 1997-07-01`)
* **Ground Truth Recurring Proportion:** 58.04% (169,475 transactions)

---

## 2. Recurring Detection Benchmark Comparison

| Metric | Heuristic Rule Baseline | Random Forest Classifier (ML) | Target / Requirement | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Precision** | **1.0000** | **0.9979** | $\ge 0.85$ | ✅ PASSED |
| **Recall** | 0.5763 | **0.9997** | $\ge 0.85$ | ✅ PASSED |
| **$F_1$-Score** | 0.7312 | **0.9988** | $\ge 0.90$ | ✅ **EXCEEDED** |
| **PR-AUC** | N/A | **1.0000** | $\ge 0.90$ | ✅ **EXCEEDED** |
| **ROC-AUC** | N/A | **1.0000** | $\ge 0.95$ | ✅ **EXCEEDED** |
| **Inference Throughput** | 114,157,251 tx/sec | 663,971 tx/sec | $> 5,000$ tx/sec | ✅ PASSED |

### Confusion Matrix Breakdown

#### Heuristic Rule Baseline:
* **True Negatives (TN):** 122,502
* **False Positives (FP):** 3
* **False Negatives (FN):** 71,815 *(missed recurring subscriptions)*
* **True Positives (TP):** 97,660

#### Random Forest ML Model:
* **True Negatives (TN):** 122,143
* **False Positives (FP):** 362
* **False Negatives (FN):** 56
* **True Positives (TP):** 169,419

> **Key Insight:** The Heuristic Rule Baseline achieved perfect precision (1.0000) but failed on recall (0.5763), missing **71,815 recurring payments**. The Machine Learning model learned complex boundary conditions and recovered **71,759 missed subscriptions**, boosting recall to **99.97%** and $F_1$ to **0.9988**.

---

## 3. Real-Time API Latency Benchmark (SLA Verification)
Evaluated across 1,000 sequential single-transaction scoring requests:

| Benchmark Metric | Measured Latency | Blueprint Target SLA | Result |
| :--- | :---: | :---: | :---: |
| **Mean Latency** | `41.46 ms` | $< 200.00\text{ ms}$ | ✅ SLA Met |
| **p50 (Median)** | `39.91 ms` | $< 200.00\text{ ms}$ | ✅ SLA Met |
| **p95** | `61.65 ms` | $< 200.00\text{ ms}$ | ✅ SLA Met |
| **p99** | `85.23 ms` | $< 200.00\text{ ms}$ | ✅ SLA Met |

---

## 4. Anomaly Detection (Isolation Forest)
* **Contamination Hyperparameter:** 0.02 (2.0%)
* **98th-Percentile Anomaly Threshold:** `0.6262`
* **Flagged Outlier Transactions:** `6,059` (2.08%)
