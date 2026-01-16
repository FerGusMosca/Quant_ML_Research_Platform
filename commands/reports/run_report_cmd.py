"""
run_report_cmd.py
-----------------
Command-line entry point to trigger ML or portfolio reports.
Handles parsing of user-provided parameters and delegates execution
to the appropriate orchestration logic layer.
"""

import sys
import time
import traceback

import os, sys

from common.enums.folders import Folders
from common.util.tagging.tagging_config_dto import TaggingConfigDTO
from logic_layer.reports_orchestration_logic import ReportsOrchestationLogic
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

Folders.load_from_config()

# ============================================================
# #2 - Report Logic Bridge
# ============================================================

def process_run_report_logic(report_key, year=None,quarter=None, portfolio=None, symbol=None, d_from=None,source=None,dest_folder=None,rank_folder=None,
                            mcp_server=None,mcp_port=None,query=None,tag_cfg=None):
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
            trd_algos.process_run_report(report_key, year, quarter=quarter, portfolio= portfolio,symbol= symbol, d_from= d_from , source=source,
                                         dest_folder= dest_folder,rank_folder= rank_folder,query=query,tag_cfg=tag_cfg)

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
    quarter = ParamReader.get_param(cmd, "quarter", True, None)
    d_from = ParamReader.get_param(cmd, "from", True, None)
    portfolio = ParamReader.get_param(cmd, "portfolio",True,None)
    source = ParamReader.get_param(cmd, "source", True, None)
    dest_folder = ParamReader.get_param(cmd, "dest_folder",True,None)
    rank_folder= ParamReader.get_param(cmd, "rank_folder",True,None)
    symbol = ParamReader.get_param(cmd, "symbol", True, None)
    server = ParamReader.get_param(cmd, "mcp_server", True, None)
    port = ParamReader.get_param(cmd, "mcp_port", True, None)
    query = ParamReader.get_param(cmd, "query", True, None)

    tag_model = ParamReader.get_param(cmd, "tag_model", True, None)
    tag_file = ParamReader.get_param(cmd, "tag_file", True, None)
    tags_csv = ParamReader.get_param(cmd, "tags_csv", True, None)
    sim_threshold = ParamReader.get_param(cmd, "sim_threshold", True, 0.8)
    doc_type = ParamReader.get_param(cmd, "doc_type", True, None)
    tag_json = ParamReader.get_param(cmd, "tag_json", True, None)

    tag_cfg=None
    if tag_model is not None:

        tag_cfg = TaggingConfigDTO(
            tag_model=tag_model,
            tag_file=tag_file,
            tags_csv=tags_csv,
            sim_threshold=sim_threshold,
            doc_type=doc_type,
            tag_json=tag_json
        )

    process_run_report_logic(report_key, year= year,quarter=quarter, portfolio=portfolio,symbol= symbol,d_from= d_from,source=source,dest_folder= dest_folder,
                             rank_folder= rank_folder, mcp_server=server,mcp_port=port,query=query,tag_cfg=tag_cfg)


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
    print("#1  StartMCP")
    print("#2  RunReport report=<report>> <portfolio> <dest_folder> <year>")
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
            f"[MCP] Starting MCP Server on server={server} port {port}",
            MessageType.INFO
        )

        loader = MLSettingsLoader()
        config = loader.load_settings("./configs/commands_mgr.ini")


        orch = ReportsOrchestationLogic(
            hist_data_conn_str= config["hist_data_conn_str"],
            ml_reports_conn_str= config["ml_reports_conn_str"],
            mcp_server= server,
            mcp_port= port,
            p_classification_map_key= None,
            logger= logger
        )

        orch._run_start_mcp()

        logger.do_log("✅ MCP Server successfully started", MessageType.INFO)

    except Exception as e:
        print(traceback.format_exc())
        logger.do_log(f"[MCP] ❌ Error: {str(e)}", MessageType.ERROR)


# ============================================================================
# DIRECT COMMAND ENTRY
# ============================================================================


def process_start_mcp(cmd):
    server = ParamReader.get_param(cmd, "mcp_server")
    port = ParamReader.get_param(cmd, "mcp_port")
    process_start_mcp_logic(server,port)


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

    print(">>> __main__ ENTERED", flush=True)
    print(f">>> sys.argv = {sys.argv}", flush=True)
    print(f">>> len(sys.argv) = {len(sys.argv)}", flush=True)

    # (A) External invocation (no menu)
    if len(sys.argv) > 1:
        print(">>> BRANCH A: external invocation", flush=True)

        cmd = " ".join(sys.argv[1:])
        print(f">>> cmd BUILT = '{cmd}'", flush=True)

        print(">>> CHECK: cmd.startswith('start_mcp') ?", flush=True)
        print(f">>> RESULT = {cmd.startswith('StartMCP')}", flush=True)

        if cmd.startswith("start_mcp"):
            print(">>> ENTERED start_mcp BRANCH", flush=True)

            print(">>> CALLING process_start_mcp(cmd)", flush=True)
            process_start_mcp(cmd)

            print(">>> ENTERING KEEP-ALIVE LOOP", flush=True)
            while True:
                print(">>> MCP KEEP-ALIVE TICK", flush=True)
                time.sleep(60)

        print(">>> ABOUT TO sys.exit(0)", flush=True)
        sys.exit(0)

    # (B) Interactive menu mode
    print(">>> BRANCH B: interactive menu", flush=True)

    while True:
        print(">>> SHOWING COMMANDS", flush=True)
        show_commands()

        print(">>> WAITING FOR INPUT()", flush=True)
        cmd = input("Enter a command: ")

        print(f">>> USER INPUT = '{cmd}'", flush=True)

        if not process_menu(cmd):
            print(">>> process_menu returned False -> BREAK", flush=True)
            break

    print(">>> RAG ingestion module closed.", flush=True)

