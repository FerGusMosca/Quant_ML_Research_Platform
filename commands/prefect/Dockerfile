FROM python:3.10-slim

WORKDIR /app

COPY commands/prefect/requirements.txt ./requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY commands/prefect /app
COPY flows /app/flows

CMD ["prefect", "server", "start"]
