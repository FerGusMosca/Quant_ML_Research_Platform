# ===== Corpus Metadata Engine =====
# All comments MUST be in English.

FROM python:3.10-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy project
COPY . /app

# Install Python deps
RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "-u", "commands/rag_corpus_metadata/run_corpus_metadata_cmd.py"]
