FROM python:3.10-slim

WORKDIR /app

# Install Prefect
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Prefect UI + API runs on 4200
EXPOSE 4200

CMD ["prefect", "server", "start"]
