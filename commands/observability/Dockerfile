# Use a slim image to keep it lightweight
FROM python:3.11-slim
WORKDIR /app
ENV PYTHONPATH=/app

# Install dependencies first for caching layers
COPY commands/observability/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy common logic and configs to allow MLSettingsLoader to work
COPY commands /app/commands
COPY common /app/common
COPY logic_layer /app/logic_layer
COPY data_access_layer /app/data_access_layer
COPY service_layer /app/service_layer
COPY business_entities /app/business_entities
COPY framework /app/framework
COPY configs /app/configs
COPY static/dictionaries /app/static/dictionaries
COPY static/tags /app/static/tags

# Expose FastAPI default port
EXPOSE 8000

# Start the proxy server
CMD ["python", "-u", "-m", "commands.observability.observability_proxy", "port=7003", "host=0.0.0.0"]

