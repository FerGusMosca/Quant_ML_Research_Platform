"""
run_corpus_metadata_cmd.py
--------------------------
Entry point for the ZEROHEDGE CORPUS METADATA pipeline.

Supports:
 - Interactive menu (full console-style)
 - Direct external command execution
"""

import sys
import traceback

from common.util.std_in_out.param_reader import ParamReader
from common.util.std_in_out.ml_settings_loader import MLSettingsLoader
from common.util.logging.logger import Logger
from framework.common.logger.message_type import MessageType
from logic_layer.corpus_metadata_orchestration_logic import CorpusMetadataOrchestrationLogic


# ============================================================================
# MENU
# ============================================================================

def show_corpus_metadata_commands():
    print("================================================================")
    print("==================== CORPUS METADATA ENGINE ====================")
    print("#1  RunCorpusMetadata mode=full  source='C:\\zerohedge_docs\\Archives' dest_root='Archives'")
    print("#2  RunCorpusMetadata mode=incremental source=<PATH> dest_root=<ROOT>")
    print("#X  Exit")
    print("================================================================")


# ============================================================================
# CORE LOGIC
# ============================================================================

def process_corpus_metadata_logic(mode, source, dest_root):
    """
    Loads configs → initializes orchestration → runs metadata pipeline.
    """
    logger = Logger()

    try:
        logger.do_log(
            f"[CORPUS] 🚀 Starting metadata mode={mode}, source={source}",
            MessageType.INFO
        )

        loader = MLSettingsLoader()
        config = loader.load_settings("./configs/commands_mgr.ini")

        orch = CorpusMetadataOrchestrationLogic(config, logger)
        orch.run(source, dest_root)

        logger.do_log("[CORPUS] ✅ Metadata generation completed.", MessageType.INFO)

    except Exception as e:
        print(traceback.format_exc())
        logger.do_log(f"[CORPUS] ❌ Error: {str(e)}", MessageType.ERROR)


# ============================================================================
# DIRECT COMMAND ENTRY
# ============================================================================

def process_corpus_metadata(cmd):
    """
    Entry point for external calls.
    Example:
        python run_corpus_metadata_cmd.py RunCorpusMetadata mode=full source=... dest_root=...
    """

    mode = ParamReader.get_param(cmd, "mode", True, None)
    source = ParamReader.get_param(cmd, "source", True, None)
    dest_root = ParamReader.get_param(cmd, "dest_root", True, None)

    process_corpus_metadata_logic(mode, source, dest_root)


# ============================================================================
# MENU DISPATCHER
# ============================================================================

def process_corpus_metadata_menu(cmd):
    tokens = cmd.split(" ")

    if tokens[0] == "RunCorpusMetadata":
        process_corpus_metadata(cmd)

    elif tokens[0].upper() in ("X", "EXIT"):
        print("Exiting CORPUS metadata module...")
        return False

    else:
        print(f"❌ Unknown command: {tokens[0]}")

    return True


# ============================================================================
# MAIN LOOP
# ============================================================================

if __name__ == "__main__":

    # (A) Direct invocation
    if len(sys.argv) > 1:
        cmd = " ".join(sys.argv[1:])
        process_corpus_metadata(cmd)
        sys.exit(0)

    # (B) Interactive menu
    while True:
        show_corpus_metadata_commands()
        cmd = input("Enter a CORPUS command: ").strip()

        if not process_corpus_metadata_menu(cmd):
            break

    print("CORPUS metadata module closed.")
