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

def process_run_report_logic(report_key, year=None, portfolio=None, symbol=None, d_from=None):
    """
    Core logic responsible for running reports through AlgosOrchestationLogic.
    """


    logger = Logger()

    try:
        logger.do_log(f"[REPORT] Starting execution for {report_key}, year={year}, portfolio={portfolio}", MessageType.INFO)

        loader = MLSettingsLoader()
        config_settings = loader.load_settings("./configs/commands_mgr.ini")

        trd_algos = ReportsOrchestationLogic(
            config_settings["hist_data_conn_str"],
            config_settings["ml_reports_conn_str"],
            None,
            logger
        )

        trd_algos.process_run_report(report_key, year, portfolio, symbol, d_from)

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
    portfolio = ParamReader.get_param(cmd, "portfolio")
    symbol = ParamReader.get_param(cmd, "symbol", True, None)

    process_run_report_logic(report_key, year, portfolio, symbol, d_from)


# ============================================================
# #4 - Script Entry
# ============================================================

if __name__ == "__main__":

    Folders.load_from_config("./configs/commands_mgr.ini")
    cmd = " ".join(sys.argv[1:])
    process_run_report(cmd)
