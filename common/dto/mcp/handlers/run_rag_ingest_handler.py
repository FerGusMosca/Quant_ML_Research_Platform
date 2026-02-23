import uuid
import asyncio
from common.dto.mcp.mcp_log_observer import MCPLogObserver


def run_rag_ingest_handler(args: dict, orchestrator):
    job_id = str(uuid.uuid4())

    logger = orchestrator.logger
    bus = orchestrator.progress_bus

    observer = MCPLogObserver(bus)
    logger.register_observer(observer)

    loop = asyncio.get_running_loop()

    def _run():
        try:
            orchestrator.process_rag_ingest(
                ingest_type=args["mode"],
                source_path=args.get("source"),
                chunk_name=args.get("chunk_name"),
                dest_root=args.get("dest_root"),
                log_posfix=args.get("log_posfix"),
                embedding_model=args.get("embedding_model"),
                clustering_model=args.get("clustering_model"),
                persist_qdrant=args.get("persist_qdrant"),
                qdrant_collection=args.get("qdrant_collection"),
                job_id=job_id,
            )
        finally:
            logger.unregister_observer(observer)

    loop.run_in_executor(None, _run)

    return {"job_id": job_id, "status": "accepted"}
