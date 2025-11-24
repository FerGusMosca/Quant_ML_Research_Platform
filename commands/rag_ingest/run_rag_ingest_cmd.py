"""
run_rag_ingest_cmd.py
---------------------
Entry point for the RAG ingestion pipeline.
Supports:
 - Interactive menu (like your main console)
 - Direct command execution from external processes
"""

import sys
import traceback

from common.util.std_in_out.param_reader import ParamReader
from common.util.std_in_out.ml_settings_loader import MLSettingsLoader
from common.util.logging.logger import Logger
from framework.common.logger.message_type import MessageType
from logic_layer.rag_ingest_orchestration_logic import RAGIngestOrchestrationLogic


# ============================================================================
# MENU
# ============================================================================

def show_rag_ingest_commands():
    print("================================================================")
    print("======================= RAG INGESTION ==========================")
    print("#1  RunRAGIngest mode=incremental source='C:\\zerohedge_docs\\Archives\\2025\\November\\Nov 6' dest_root='Archives'")
    print("#2  RunRAGIngest mode=full source=<PATH>>")
    print("#X  Exit")
    print("================================================================")


# ============================================================================
# CORE LOGIC
# ============================================================================

def process_rag_ingest_logic(mode, source,dest_root):
    """
    Core logic runner for ingestion.
    Loads config → creates orchestrator → runs selected pipeline.
    """

    logger = Logger()

    try:
        logger.do_log(
            f"[RAG] Starting ingestion mode={mode}, source={source}",
            MessageType.INFO
        )

        loader = MLSettingsLoader()
        config = loader.load_settings("./configs/commands_mgr.ini")

        orch = RAGIngestOrchestrationLogic(config, logger)
        orch.process_rag_ingest(mode, source,dest_root)

        logger.do_log("[RAG] ✅ Ingestion completed", MessageType.INFO)

    except Exception as e:
        print(traceback.format_exc())
        logger.do_log(f"[RAG] ❌ Error: {str(e)}", MessageType.ERROR)


# ============================================================================
# DIRECT COMMAND ENTRY
# ============================================================================

def process_rag_ingest(cmd):
    """
    External entrypoint.
    Example external call:
        python run_rag_ingest_cmd.py RunRAGIngest mode=full source=ZEROHEDGE
    """

    mode = ParamReader.get_param(cmd, "mode")
    source = ParamReader.get_param(cmd, "source", True, None)
    dest_root = ParamReader.get_param(cmd, "dest_root", True, None)

    process_rag_ingest_logic(mode, source,dest_root)


# ============================================================================
# MENU HANDLER
# ============================================================================

def process_rag_ingest_menu(cmd):
    """
    Menu dispatcher (mirrors your big console pattern).
    """

    tokens = cmd.split(" ")

    if tokens[0] == "RunRAGIngest":
        process_rag_ingest(cmd)

    elif tokens[0].upper() in ("EXIT", "X"):
        print("Exiting RAG ingestion module...")
        return False

    else:
        print(f"❌ Unknown command: {tokens[0]}")

    return True


# ============================================================================
# MAIN LOOP
# ============================================================================

if __name__ == "__main__":

    # (A) External invocation (no menu)
    # ---------------------------------------------------
    # Example:
    # python run_rag_ingest_cmd.py RunRAGIngest mode=incremental source=ZEROHEDGE
    if len(sys.argv) > 1:
        cmd = " ".join(sys.argv[1:])
        process_rag_ingest(cmd)
        sys.exit(0)

    # (B) Interactive menu mode
    # ---------------------------------------------------
    while True:
        show_rag_ingest_commands()
        cmd = input("Enter a RAG command: ")

        if not process_rag_ingest_menu(cmd):
            break

    print("RAG ingestion module closed.")
