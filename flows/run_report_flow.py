from prefect import flow, task
import subprocess

@task
def run_raw_command(cmd: str):
    subprocess.run(
        ["python", "commands/reports/run_report_cmd.py"] + cmd.split(" "),
        check=True
    )

@flow
def run_report_flow(cmd: str):
    run_raw_command(cmd)
