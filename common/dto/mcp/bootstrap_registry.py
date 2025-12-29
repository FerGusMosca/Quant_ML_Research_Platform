# service_layer/mcp/bootstrap_registry.py
from common.dto.mcp.handlers.run_report_handler import run_report_handler
from common.dto.mcp.tools import ToolRegistry, Tool, ToolSpec


def build_mcp_registry(orchestrator) -> ToolRegistry:
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
                        "year": {"type": "integer"},
                        "portfolio": {"type": "string"},
                        "symbol": {"type": "string"},
                        "d_from": {"type": "string"},
                        "dest_folder": {"type": "string"},
                        "rank_folder": {"type": "string"},
                    },
                    "required": ["report"],
                },
            ),
            handler=lambda args: run_report_handler(args, orchestrator),
        )
    )

    return registry
