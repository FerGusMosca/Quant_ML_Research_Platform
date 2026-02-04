# service_layer/mcp/handlers/run_report_handler.py
import uuid
import asyncio
from common.dto.mcp.mcp_log_observer import MCPLogObserver
from common.util.tagging.tagging_config_dto import TaggingConfigDTO


def run_report_handler(args: dict, orchestrator):
    job_id = str(uuid.uuid4())

    logger = orchestrator.logger
    bus = orchestrator.progress_bus

    observer = MCPLogObserver(bus)
    logger.register_observer(observer)

    loop = asyncio.get_running_loop()

    def _run():
        try:

            tag_cfg=None
            if args.get("tag_model") is not None:
                tag_cfg = TaggingConfigDTO(
                    tag_model=args.get("tag_model"),
                    tag_file=args.get("tag_file"),
                    tag_json=args.get("tag_json"),
                    tags_csv=None,
                    sim_threshold=None,
                    doc_type=args.get("doc_type"),
                    tag_dedup=bool(args.get("tag_dedup"))
                )


            orchestrator.process_run_report(
                report_key=args["report"],
                year=args.get("year"),
                quarter=args.get("quarter"),
                portfolio=args.get("portfolio"),
                symbol=args.get("symbol"),
                d_from=args.get("d_from"),
                source=args.get("source"),
                dest_folder=args.get("dest_folder"),
                rank_folder=args.get("rank_folder"),
                tag_cfg=tag_cfg,
                job_id=job_id,
            )
        except Exception as e:
            print(f"Error initializing process_run_report: {str(e)} ")
        finally:
            logger.unregister_observer(observer)

    loop.run_in_executor(None, _run)

    return {"job_id": job_id, "status": "accepted"}

