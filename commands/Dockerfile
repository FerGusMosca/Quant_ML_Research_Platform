# --------------------------------------------
# Dockerfile for run_report_cmd
# --------------------------------------------
FROM python:3.10-slim

WORKDIR /app

# Copy only what we need (evita peso innecesario)
COPY . /app

# Install dependencies
RUN pip install --upgrade pip && pip install -r commands/requirements.txt

# Config file path (mountable)
ENV COMMANDS_MGR_PATH="/deploys/commands/commands_mgr.ini"

# Entrypoint
ENTRYPOINT ["python", "-m", "commands.run_report_cmd"]
