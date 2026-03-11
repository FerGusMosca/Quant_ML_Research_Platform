# service_layer/mcp/bootstrap_registry.py
from common.dto.mcp.handlers.run_corpus_metadata_handler import run_corpus_metadata_handler
from common.dto.mcp.handlers.run_rag_ingest_handler import run_rag_ingest_handler
from common.dto.mcp.handlers.run_report_handler import run_report_handler
from common.dto.mcp.tools import ToolRegistry, Tool, ToolSpec


def build_mcp_registry_reports(orchestrator) -> ToolRegistry:
    registry = ToolRegistry()

    registry.register(
        Tool(
            spec=ToolSpec(
                name="run_report",
                description="Execute report orchestration pipeline",
                input_schema={
                    "type": "object",
                    "properties": {
                        "report": {"type": "string"},
                        "year": {"type": "string"},
                        "quarter": {"type": "string"},
                        "portfolio": {"type": "string"},
                        "symbol": {"type": "string"},
                        "d_from": {"type": "string"},
                        "dest_folder": {"type": "string"},
                        "rank_folder": {"type": "string"},
                        "tag_model": {"type": "string"},
                        "tag_json": {"type": "string"},
                        "tag_file": {"type": "string"},
                        "tag_dedup": {"type": "boolean"},
                        "doc_type": {"type": "string"},
                    },
                    "required": ["report"],
                },
            ),
            handler=lambda args: run_report_handler(args, orchestrator),
        )
    )

    return registry

# -------------------------
# run_rag_ingest (NEW)
# -------------------------
def build_mcp_registry_ingest(orchestrator) -> ToolRegistry:
    registry = ToolRegistry()

    registry.register(
        Tool(
            spec=ToolSpec(
                name="run_rag_ingest",
                description="Run RAG ingestion pipeline (full / incremental)",
                input_schema={
                    "type": "object",
                    "properties": {
                        "mode": {"type": "string"},
                        "source": {"type": "string"},
                        "dest_root": {"type": "string"},
                        "chunk_name": {"type": "string"},
                        "log_posfix": {"type": "string"},
                        "embedding_model": {"type": "string"},
                        "clustering_model": {"type": "string"},
                        "persist_qdrant": {"type": "boolean"},
                        "qdrant_collection": {"type": "string"},
                    },
                    "required": ["mode", "source", "dest_root", "chunk_name"],
                },
            ),
            handler=lambda args: run_rag_ingest_handler(args, orchestrator),
        )
    )

    return registry


def build_mcp_registry_corpus_metadata(registry,orchestrator) -> ToolRegistry:
    #registry = ToolRegistry()

    registry.register(
        Tool(
            spec=ToolSpec(
                name="run_corpus_metadata",
                description="Run corpus metadata generation pipeline (full / incremental)",
                input_schema={
                    "type": "object",
                    "properties": {
                        "mode":       {"type": "string"},
                        "source":     {"type": "string"},
                        "dest_root":  {"type": "string"},
                        "chunk_name": {"type": "string"},
                    },
                    "required": ["source", "dest_root", "chunk_name"],
                },
            ),
            handler=lambda args: run_corpus_metadata_handler(args, orchestrator),
        )
    )

    return registry