# Multi-stage Dockerfile for FastAPI microservice and Streamlit dashboard
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency specifications
COPY requirements.txt .

# Install python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and artifacts
COPY config/ ./config/
COPY src/ ./src/
COPY app/ ./app/
COPY models/ ./models/
COPY README.md .

# Expose ports: 8000 (FastAPI), 8501 (Streamlit)
EXPOSE 8000
EXPOSE 8501

# Default launch command: runs FastAPI microservice
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
