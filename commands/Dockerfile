FROM python:3.10-slim

WORKDIR /app

# 🔧 Install ODBC system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    unixodbc \
    unixodbc-dev \
    && rm -rf /var/lib/apt/lists/*

COPY ./commands/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY ./commands /app/commands
COPY ./common /app/common
COPY ./logic_layer /app/logic_layer
COPY ./data_access_layer /app/data_access_layer
COPY ./framework /app/framework
COPY ./configs /app/configs

CMD ["python", "-m", "commands.run_report_cmd", "RunReport"]
