FROM python:3.10-slim

WORKDIR /app
ENV PYTHONPATH=/app

# ------------------ Sistema ------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl gnupg apt-transport-https ca-certificates \
    chromium chromium-driver \
    unixodbc-dev && \
    # ODBC Driver 18 for SQL Server
    curl -fsSL https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor -o /usr/share/keyrings/microsoft-prod.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/microsoft-prod.gpg] https://packages.microsoft.com/debian/12/prod bookworm main" > /etc/apt/sources.list.d/mssql-release.list && \
    apt-get update && \
    ACCEPT_EULA=Y apt-get install -y --no-install-recommends msodbcsql18 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# ------------------ Python deps ------------------
COPY ./commands/reports/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ------------------ NLTK data (precargar todo lo que usa tu código) ------------------
RUN python -c "import nltk; nltk.download('vader_lexicon', quiet=True); nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)"

# ------------------ Spacy model ------------------
RUN python -m spacy download en_core_web_sm

# ------------------ Código y recursos ------------------
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

# ------------------ Command ------------------
CMD ["python", "-u", "-m", "commands.reports.run_report_cmd"]