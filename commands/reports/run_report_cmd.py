"""
run_report_cmd.py
-----------------
Command-line entry point to trigger ML or portfolio reports.
Handles parsing of user-provided parameters and delegates execution
to the appropriate orchestration logic layer.
"""

import sys
import traceback

from common.enums.folders import Folders
from common.util.financial_calculations.date_handler import DateHandler
from common.util.logging.logger import Logger
from common.util.std_in_out.ml_settings_loader import MLSettingsLoader
from common.util.std_in_out.param_reader import ParamReader
from framework.common.logger.message_type import MessageType

import os, sys

from logic_layer.reports_orchestration_logic import ReportsOrchestationLogic

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# ============================================================
# #2 - Report Logic Bridge
# ============================================================

def process_run_report_logic(report_key, year=None, portfolio=None, symbol=None, d_from=None,dest_folder=None,rank_folder=None,
                            mcp_server=None,mcp_port=None):
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
            trd_algos.process_run_report(report_key, year, portfolio, symbol, d_from,dest_folder,rank_folder)

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

    process_run_report_logic(report_key, year, portfolio, symbol, d_from, dest_folder, rank_folder, mcp_server=server,
                             mcp_port=port)


# ============================================================
# #4 - Script Entry
# ============================================================

if __name__ == "__main__":

    Folders.load_from_config("./configs/commands_mgr.ini")
    cmd = " ".join(sys.argv[1:])
    process_run_report(cmd)
