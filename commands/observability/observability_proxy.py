"""
Observability Proxy Server
==========================
FastAPI service that receives observability events from MCP services
and routes them to Langfuse via the OrchestrationLogic layer.
"""

import os
import sys

import uvicorn
from fastapi import FastAPI, Body

# Standard path setup (BEFORE other imports)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT_DIR)

from common.util.std_in_out.ml_settings_loader import MLSettingsLoader
from common.util.std_in_out.param_reader import ParamReader
from logic_layer.observability_orchestation_logic import ObservabilityOrchestationLogic


app = FastAPI(
    title="Observability Proxy",
    description="Routes observability events from MCP services to Langfuse",
    version="1.0.0"
)

# ============================================================
# INITIALIZE ORCHESTRATOR AT MODULE LEVEL
# This ensures it exists when uvicorn imports the module
# ============================================================
loader = MLSettingsLoader()
config_settings = loader.load_settings("./configs/commands_mgr.ini")
orchestrator = ObservabilityOrchestationLogic(config=config_settings)


@app.post("/log")
async def log_event(data: dict = Body(...)):
    """
    Unified entry point for decoupled logging.
    Delegates execution to the ObservabilityOrchestationLogic.
    """
    try:
        orchestrator.log_process_step(
            step_name=data.get("node_name"),
            input_data=data.get("input"),
            output_data=data.get("output"),
            trace_id=data.get("trace_id"),
            trace_name=data.get("trace_name", "mcp-process"),
            level=data.get("level", "DEFAULT"),
            parent_id=data.get("parent_id"),
            service_id=data.get("service_id")
        )
        return {"status": "ok"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "orchestrator": orchestrator is not None}


# ============================================================
# BOOTSTRAP ENTRY POINT (only for local development)
# ============================================================
if __name__ == "__main__":
    cmd_args = " ".join(sys.argv)

    listen_port = int(ParamReader.get_param(cmd_args, "port", True, 7003))
    listen_host = ParamReader.get_param(cmd_args, "host", True, "0.0.0.0")

    print("=" * 64)
    print("================= OBSERVABILITY PROXY =========================")
    print(f">>> Status: RUNNING")
    print(f">>> Listen Address: {listen_host}:{listen_port}")
    print("=" * 64)

    uvicorn.run(app, host=listen_host, port=listen_port)