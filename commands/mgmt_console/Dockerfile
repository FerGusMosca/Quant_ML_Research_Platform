FROM python:3.10-slim

WORKDIR /app

# ------------------------------------------------------
# Add ODBC Driver for SQL Server (Debian 12 - FINAL FIX)
# ------------------------------------------------------
RUN apt-get update && \
    apt-get install -y curl gnupg apt-transport-https ca-certificates && \
    mkdir -p /etc/apt/keyrings && \
    curl -fsSL https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor -o /etc/apt/keyrings/microsoft.gpg && \
    echo "deb [arch=amd64,arm64 signed-by=/etc/apt/keyrings/microsoft.gpg] https://packages.microsoft.com/debian/12/prod bookworm main" > /etc/apt/sources.list.d/mssql-release.list && \
    apt-get update && \
    ACCEPT_EULA=Y apt-get install -y msodbcsql18 unixodbc-dev git && \
    apt-get install -y libgl1 libglib2.0-0 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------
# Copy requirements file
# ------------------------------------------------------
COPY ./commands/mgmt_console/requirements.txt ./requirements.txt

# ------------------------------------------------------
# Install Python dependencies (order critical)
# ------------------------------------------------------
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir tradingview-datafeed && \
    pip install --no-cache-dir -r requirements.txt && \
    pip check

# ------------------------------------------------------
# Copy app source (solo módulos necesarios)
# ------------------------------------------------------
COPY ./common /app/common
COPY ./configs /app/configs
COPY ./controllers /app/controllers
COPY ./data_access_layer /app/data_access_layer
COPY ./framework /app/framework
COPY ./logic_layer /app/logic_layer
COPY ./service_layer /app/service_layer
COPY ./scripts /app/scripts
COPY ./commands /app/commands
COPY ./business_entities /app/business_entities
COPY ./templates /app/templates
COPY ./POC /app/POC
COPY ./models /app/models
COPY ./static /app/static
RUN touch /app/.project_root

# Archivo principal
COPY ./bias_mgmt_console.py /app/bias_mgmt_console.py
COPY ./ui_dashboard.py /app/ui_dashboard.py

# ------------------------------------------------------
# Health check + default command
# ------------------------------------------------------
#HEALTHCHECK CMD python -c "import tvdatafeed" || exit 1
CMD ["python", "/app/bias_mgmt_console.py"]

EXPOSE 8000