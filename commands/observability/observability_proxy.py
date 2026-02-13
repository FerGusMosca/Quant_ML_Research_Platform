import os
import sys
import uvicorn
from fastapi import FastAPI, Body

from common.util.std_in_out.ml_settings_loader import MLSettingsLoader
# Import your custom logic and tools
from logic_layer.observability_orchestation_logic import ObservabilityOrchestationLogic
# Note: Ensure pandas is installed in venv_observability if ParamReader is used
from common.util.std_in_out.param_reader import ParamReader

# Standard path setup
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT_DIR)



app = FastAPI()

# Global instance of the Orchestation Logic
# It will handle Langfuse initialization internally
orchestrator = None


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
            parent_id=data.get("parent_id")
        )
        return {"status": "ok"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ============================================================
# BOOTSTRAP ENTRY POINT
# ============================================================
if __name__ == "__main__":
    # Standardize args for ParamReader pattern
    cmd_args = " ".join(sys.argv)

    # Port 7003 as discussed
    listen_port = int(ParamReader.get_param(cmd_args, "port", True, 7003))
    listen_host = ParamReader.get_param(cmd_args, "host", True, "0.0.0.0")

    loader = MLSettingsLoader()
    config_settings = loader.load_settings("./configs/commands_mgr.ini")



    orchestrator = ObservabilityOrchestationLogic(config=config_settings)

    print("================================================================")
    print("=================== OBSERVABILITY PROXY ========================")
    print(f">>> Status: RUNNING")
    print(f">>> Listen Address: {listen_host}:{listen_port}")
    print("================================================================")

    uvicorn.run(app, host=listen_host, port=listen_port)