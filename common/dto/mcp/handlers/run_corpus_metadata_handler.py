import uuid
import asyncio
from common.dto.mcp.mcp_log_observer import MCPLogObserver
from common.util.observability.langfuse.observability_context import ObsContext


def run_corpus_metadata_handler(args: dict, orchestrator):
    job_id = str(uuid.uuid4())

    logger = orchestrator.logger
    bus    = orchestrator.progress_bus

    observer = MCPLogObserver(bus)
    logger.register_observer(observer)

    loop = asyncio.get_running_loop()

    def _run():
        try:
            ObsContext.set(job_id, service_id="corpus-metadata", operation_name="run_corpus_metadata")
            orchestrator.run_corpus_metadata(
                source_path=args.get("source"),
                dest_root=args.get("dest_root"),
                chunk_name=args.get("chunk_name"),
                tag_cfg=None,
                job_id=job_id
            )
        finally:
            logger.unregister_observer(observer)

    loop.run_in_executor(None, _run)

    return {"job_id": job_id, "status": "accepted"}