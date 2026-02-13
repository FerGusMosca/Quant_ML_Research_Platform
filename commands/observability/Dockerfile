# Use a slim image to keep it lightweight
FROM python:3.11-slim

WORKDIR /app

# Install dependencies first for caching layers
COPY commands/observability/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy common logic and configs to allow MLSettingsLoader to work
COPY common/ ./common/
COPY configs/ ./configs/
COPY commands/observability/observability_proxy.py .

# Expose FastAPI default port
EXPOSE 8000

# Start the proxy server
CMD ["uvicorn", "observability_proxy:app", "--host", "0.0.0.0", "--port", "8000"]