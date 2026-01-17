FROM python:3.10-slim

WORKDIR /app
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# ---------------------------------------------------
# 1) System dependencies (mínimas, runtime-only)
# ---------------------------------------------------
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
        build-essential \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------
# 2) Python dependencies
# ---------------------------------------------------
COPY ./commands/mcp_workflows/requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

# ---------------------------------------------------
# 3) Copy project files (solo lo necesario)
# ---------------------------------------------------
COPY ./commands /app/commands
COPY ./common /app/common
COPY ./service_layer /app/service_layer
COPY ./configs /app/configs

# ---------------------------------------------------
# 4) Default entrypoint (neutro)
# ---------------------------------------------------
ENTRYPOINT ["python", "-u"]
