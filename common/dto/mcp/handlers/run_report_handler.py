# service_layer/mcp/handlers/run_report_handler.py
import uuid
import asyncio
from common.dto.mcp.mcp_log_observer import MCPLogObserver


def run_report_handler(args: dict, orchestrator):
    job_id = str(uuid.uuid4())

    logger = orchestrator.logger
    bus = orchestrator.progress_bus

    observer = MCPLogObserver(bus)
    logger.register_observer(observer)

    loop = asyncio.get_running_loop()

    def _run():
        try:
            orchestrator.process_run_report(
                report_key=args["report"],
                year=args.get("year"),
                portfolio=args.get("portfolio"),
                symbol=args.get("symbol"),
                d_from=args.get("d_from"),
                dest_folder=args.get("dest_folder"),
                rank_folder=args.get("rank_folder"),
                job_id=job_id,
            )
        finally:
            logger.unregister_observer(observer)

    loop.run_in_executor(None, _run)

    return {"job_id": job_id, "status": "accepted"}

