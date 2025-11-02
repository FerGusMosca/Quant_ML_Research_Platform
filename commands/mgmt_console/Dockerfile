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
    ACCEPT_EULA=Y apt-get install -y msodbcsql18 unixodbc-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------
# Install dependencies
# ------------------------------------------------------
COPY ./commands/mgmt_console/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ------------------------------------------------------
# Copy app source
# ------------------------------------------------------
COPY ./commands/mgmt_console /app/commands/mgmt_console
COPY ./common /app/common
COPY ./configs /app/configs
COPY ./logic_layer /app/logic_layer
COPY ./data_access_layer /app/data_access_layer
COPY ./framework /app/framework

# ------------------------------------------------------
# Default command
# ------------------------------------------------------
CMD ["python", "/app/bias_mgmt_console.py"]