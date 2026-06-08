# Dockerfile — containerised deployment of the prediction API.
#
# Build:  docker build -t blr-rental-predictor .
# Run:    docker run -p 8000:8000 blr-rental-predictor
# Then:   http://localhost:8000/docs

FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code + trained model
COPY api.py .
COPY src/ ./src/
COPY outputs/xgb_model.pkl outputs/scaler.pkl ./outputs/

EXPOSE 8000

# Use shell form so $PORT (if set by host) is respected; default 8000
CMD uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000}
