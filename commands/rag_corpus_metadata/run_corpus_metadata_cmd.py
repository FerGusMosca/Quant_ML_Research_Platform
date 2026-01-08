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
from common.util.tagging.tagging_config_dto import TaggingConfigDTO
from framework.common.logger.message_type import MessageType
from logic_layer.corpus_metadata_orchestration_logic import CorpusMetadataOrchestrationLogic


# ============================================================================
# MENU
# ============================================================================

def show_corpus_metadata_commands():
    print("================================================================")
    print("==================== CORPUS METADATA ENGINE ====================")
    print("#1  RunCorpusMetadata mode=incremental source=<PATH> dest_root=<ROOT> tag_model=<sentence-transformers/all-MiniLM-L6-v2> tag_file=<KQ10_tags.json>")
    print("#X  Exit")
    print("================================================================")


# ============================================================================
# CORE LOGIC
# ============================================================================

def process_corpus_metadata_logic(mode, source, dest_root,chunk_name,tag_cfg=None):
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
        orch.run(source, dest_root,chunk_name,tag_cfg=tag_cfg)

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
    chunk_name = ParamReader.get_param(cmd, "chunk_name", True, None)
    tag_model = ParamReader.get_param(cmd, "tag_model", True, None)
    tag_file = ParamReader.get_param(cmd, "tag_file", True, None)
    tags_csv = ParamReader.get_param(cmd, "tags_csv", True, None)
    sim_threshold = ParamReader.get_param(cmd, "sim_threshold", True, 0.8)
    doc_type = ParamReader.get_param(cmd, "doc_type", True, None)

    tag_cfg=None
    if tag_model is not None:

        tag_cfg = TaggingConfigDTO(
            tag_model=tag_model,
            tag_file=tag_file,
            tags_csv=tags_csv,
            sim_threshold=sim_threshold,
            doc_type=doc_type
        )


    process_corpus_metadata_logic(mode, source, dest_root,chunk_name,tag_cfg=tag_cfg)


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
