FROM python:3.10-slim

WORKDIR /app

# ------------------------------------------------------
# Add ODBC Driver for SQL Server
# ------------------------------------------------------
RUN apt-get update && \
    apt-get install -y curl apt-transport-https gnupg && \
    curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add - && \
    curl https://packages.microsoft.com/config/debian/11/prod.list > /etc/apt/sources.list.d/mssql-release.list && \
    apt-get update && \
    ACCEPT_EULA=Y apt-get install -y msodbcsql18 unixodbc-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------
# Install dependencies
# ------------------------------------------------------
COPY ./commands/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ------------------------------------------------------
# Copy app source
# ------------------------------------------------------
COPY ./commands /app/commands
COPY ./common /app/common
COPY ./logic_layer /app/logic_layer
COPY ./data_access_layer /app/data_access_layer
COPY ./framework /app/framework
COPY ./configs /app/configs

# Default command (override via docker run)
CMD ["python", "-m", "commands.run_report_cmd"]
