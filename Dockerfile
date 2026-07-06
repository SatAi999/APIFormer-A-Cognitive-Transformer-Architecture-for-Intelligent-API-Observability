# Multi-stage Dockerfile for APIFormer+ Serving and Analytics Dashboard

# Stage 1: Build Dependencies
FROM python:3.10-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Stage 2: Final Run-time Container
FROM python:3.10-slim AS runner

WORKDIR /app

# Copy installed packages from builder stage
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy project workspace
COPY config/ ./config/
COPY src/ ./src/
COPY data/processed/ ./data/processed/
COPY data/checkpoints/ ./data/checkpoints/

# Expose FastAPI serving port and Streamlit port
EXPOSE 8000
EXPOSE 8501

ENV PYTHONPATH="."

# Default command launches FastAPI predictor app
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
