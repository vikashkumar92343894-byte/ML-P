import time
from typing import List
from fastapi import FastAPI, HTTPException
from src.data.schema import TransactionInput, ScoreResult, CustomerSummary
from src.models.predict import ExpenseDetector
from src.utils.logger import get_logger

logger = get_logger("api")

app = FastAPI(
    title="Hidden Subscription & Expense Detector API",
    description="Real-time ML scoring microservice for detecting hidden subscriptions, price increases, duplicates, and anomalies.",
    version="1.0.0"
)

# Initialize detector engine singleton
detector = ExpenseDetector()

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "recurring_model_loaded": detector.recurring_model is not None,
        "anomaly_model_loaded": detector.anomaly_model is not None,
        "active_groups_in_memory": len(detector.history)
    }

@app.post("/score", response_model=ScoreResult)
def score_transaction(t: TransactionInput):
    start_time = time.perf_counter()
    result = detector.score_transaction(t)
    duration_ms = (time.perf_counter() - start_time) * 1000.0
    
    if duration_ms > 200.0:
        logger.warning(f"Latency SLA alert: scored txn in {duration_ms:.2f} ms")
    
    return result

@app.post("/score_batch", response_model=List[ScoreResult])
def score_batch(transactions: List[TransactionInput]):
    return [detector.score_transaction(t) for t in transactions]

@app.get("/customer/{customer_id}/summary", response_model=CustomerSummary)
def get_customer_summary(customer_id: str):
    return detector.get_customer_summary(customer_id)

@app.post("/reset")
def reset_history():
    detector.history.clear()
    detector.alerts_log.clear()
    return {"message": "In-memory transaction history successfully reset."}
