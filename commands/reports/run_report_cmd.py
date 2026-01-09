"""
run_report_cmd.py
-----------------
Command-line entry point to trigger ML or portfolio reports.
Handles parsing of user-provided parameters and delegates execution
to the appropriate orchestration logic layer.
"""

import sys
import traceback

import os, sys

from logic_layer.reports_orchestration_logic import ReportsOrchestationLogic
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# ============================================================
# #2 - Report Logic Bridge
# ============================================================

def process_run_report_logic(report_key, year=None, portfolio=None, symbol=None, d_from=None,dest_folder=None,rank_folder=None,
                            mcp_server=None,mcp_port=None,query=None):
    """
    Core logic responsible for running reports through AlgosOrchestationLogic.
    """


    logger = Logger()
    try:
        logger.do_log(f"[REPORT] Starting execution for {report_key}, year={year}, portfolio={portfolio}", MessageType.INFO)



        loader = MLSettingsLoader()
        config_settings = loader.load_settings("./configs/commands_mgr.ini")

        trd_algos = ReportsOrchestationLogic(
            hist_data_conn_str= config_settings["hist_data_conn_str"],
            ml_reports_conn_str= config_settings["ml_reports_conn_str"],
            mcp_server= mcp_server,
            mcp_port= mcp_port,
            p_classification_map_key= None,
            logger= logger
        )

        if mcp_port is not None and mcp_port is not None:
            logger.do_log(f"[REPORT] Detected MCP Server start commands : server={mcp_port} port={mcp_port}",
                          MessageType.INFO)
            trd_algos._run_start_mcp()
        else:
            trd_algos.process_run_report(report_key, year, portfolio, symbol, d_from,dest_folder,rank_folder,query=query)

        logger.do_log(f"[REPORT] ✅ Report {report_key} completed", MessageType.INFO)

    except Exception as e:
        print(traceback.format_exc())
        logger.do_log(f"[REPORT] ❌ Error executing report {report_key} - {str(e)}", MessageType.ERROR)



# ============================================================
# #3 - Entry Point
# ============================================================

def process_run_report(cmd):
    """
    Extracts parameters from command string and delegates to process_run_report_logic().
    Example:
        RunReport report=download_q10 portfolio=US_BIGCAP_EX year=2025
    """
    report_key = ParamReader.get_param(cmd, "report")
    year = ParamReader.get_param(cmd, "year", True, None)
    d_from = ParamReader.get_param(cmd, "from", True, None)
    portfolio = ParamReader.get_param(cmd, "portfolio",True,None)
    dest_folder = ParamReader.get_param(cmd, "dest_folder",True,None)
    rank_folder= ParamReader.get_param(cmd, "rank_folder",True,None)
    symbol = ParamReader.get_param(cmd, "symbol", True, None)
    server = ParamReader.get_param(cmd, "mcp_server", True, None)
    port = ParamReader.get_param(cmd, "mcp_port", True, None)
    query = ParamReader.get_param(cmd, "query", True, None)

    process_run_report_logic(report_key, year, portfolio, symbol, d_from, dest_folder, rank_folder, mcp_server=server,
                             mcp_port=port,query=query)


# ============================================================
# #4 - Script Entry
# ============================================================
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

def show_commands():
    print("================================================================")
    print("======================= RAG INGESTION ==========================")
    print("#1  RunRAGIngest mode=incremental source='C:\\zerohedge_docs\\Archives\\2025\\November\\Nov 6' dest_root='Archives'")
    print("#2  RunRAGIngest mode=full source=<PATH>>")
    print("#3  StartMCP")
    print("#4  RunReport report=<report>> <portfolio> <dest_folder> <year>")
    print("#X  Exit")
    print("================================================================")


# ============================================================================
# CORE LOGIC
# ============================================================================


def process_start_mcp_logic(server,port):
    """
       Core logic runner for ingestion.
       Loads config → creates orchestrator → runs selected pipeline.
       """

    logger = Logger()

    try:
        logger.do_log(
            f"[RAG] Starting MCP Server on server={server} port {port}",
            MessageType.INFO
        )

        loader = MLSettingsLoader()
        config = loader.load_settings("./configs/commands_mgr.ini")

        orch = RAGIngestOrchestrationLogic(config, logger)
        orch.process_start_mcp(server,port)

        logger.do_log("[RAG] ✅ Ingestion completed", MessageType.INFO)

    except Exception as e:
        print(traceback.format_exc())
        logger.do_log(f"[RAG] ❌ Error: {str(e)}", MessageType.ERROR)

def process_rag_ingest_logic(mode, source,chunk_name,dest_root,log_posfix,embedding_model,clustering_model):
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
        orch.process_rag_ingest(mode, source,chunk_name,dest_root,log_posfix,embedding_model,clustering_model)

        logger.do_log("[RAG] ✅ Ingestion completed", MessageType.INFO)

    except Exception as e:
        print(traceback.format_exc())
        logger.do_log(f"[RAG] ❌ Error: {str(e)}", MessageType.ERROR)


# ============================================================================
# DIRECT COMMAND ENTRY
# ============================================================================


def process_start_mcp(cmd):
    server = ParamReader.get_param(cmd, "mcp_server")
    port = ParamReader.get_param(cmd, "mcp_port")
    process_start_mcp_logic(server,port)

def process_rag_ingest(cmd):
    """
    External entrypoint.
    Example external call:
        python run_rag_ingest_cmd.py RunRAGIngest mode=full source=ZEROHEDGE
    """

    mode = ParamReader.get_param(cmd, "mode")
    source = ParamReader.get_param(cmd, "source", True, None)
    dest_root = ParamReader.get_param(cmd, "dest_root", True, None)
    chunk_name = ParamReader.get_param(cmd, "chunk_name", True, None)
    log_posfix = ParamReader.get_param(cmd, "log_posfix", True, None)
    embedding_model = ParamReader.get_param(cmd, "embedding_model", True, None)
    clustering_model = ParamReader.get_param(cmd, "clustering_model", True, None)

    process_rag_ingest_logic(mode, source,chunk_name,dest_root,log_posfix,embedding_model,clustering_model)


# ============================================================================
# MENU HANDLER
# ============================================================================

def process_menu(cmd):
    """
    Menu dispatcher (mirrors your big console pattern).
    """

    tokens = cmd.split(" ")


    if tokens[0] == "StartMCP":
        process_start_mcp(cmd)
    if tokens[0] == "RunReport":
        process_run_report(cmd)
    #
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
        if  cmd.startswith("StartMCP"):
            process_start_mcp(cmd)

        sys.exit(0)

    # (B) Interactive menu mode
    # ---------------------------------------------------
    while True:
        show_commands()
        cmd = input("Enter a command: ")

        if not process_menu(cmd):
            break

    print("RAG ingestion module closed.")
