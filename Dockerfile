# Multi-stage production Dockerfile for PartSense
FROM python:3.11-slim as base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code and pre-computed artifacts
COPY configs/ ./configs/
COPY data/ ./data/
COPY models/ ./models/
COPY reports/ ./reports/
COPY sql/ ./sql/
COPY src/ ./src/
COPY app/ ./app/
COPY pyproject.toml .
COPY .streamlit/ ./.streamlit/

# Install local package in editable mode without reinstalling dependencies
RUN pip install --no-cache-dir -e . --no-deps

# Create non-root user for security
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app/Home.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
