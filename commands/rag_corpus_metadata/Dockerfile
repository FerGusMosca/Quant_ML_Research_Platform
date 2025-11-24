# ===== Corpus Metadata Engine =====
FROM python:3.10-slim
WORKDIR /app

# 1) System deps (MISMO INGEST)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        gnupg \
        apt-transport-https \
        ca-certificates \
        build-essential \
        unixodbc-dev \
        && \
    mkdir -p /etc/apt/keyrings && \
    curl -fsSL https://packages.microsoft.com/keys/microsoft.asc \
        | gpg --dearmor -o /etc/apt/keyrings/microsoft.gpg && \
    echo "deb [arch=amd64,arm64 signed-by=/etc/apt/keyrings/microsoft.gpg] https://packages.microsoft.com/debian/12/prod bookworm main" \
        > /etc/apt/sources.list.d/mssql-release.list && \
    apt-get update && \
    ACCEPT_EULA=Y apt-get install -y msodbcsql18 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# 2) Python deps
COPY ./commands/rag_corpus_metadata/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

# 3) NLTK (idéntico ingest)
RUN python3 -m nltk.downloader punkt punkt_tab

# 4) Copy project
COPY ./commands /app/commands
COPY ./common /app/common
COPY ./logic_layer /app/logic_layer
COPY ./data_access_layer /app/data_access_layer
COPY ./framework /app/framework
COPY ./configs /app/configs

# 5) CMD
CMD ["python","-u","-m","commands.rag_corpus_metadata.run_corpus_metadata_cmd"]
