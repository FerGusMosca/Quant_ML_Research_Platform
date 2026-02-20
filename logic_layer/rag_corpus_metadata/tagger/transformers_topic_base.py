# FILE: transformers_topic_base.py
# BASE CLASS – NO LOGIC CHANGES. METHODS COPIED 1:1.

import os
import json
import traceback
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from bs4 import BeautifulSoup
import warnings
from bs4 import XMLParsedAsHTMLWarning
from common.util.extractors.K_Q_10.k_q_10_html_structured_block_extractor import KQ10HtmlStructuredBlockExtractor
from common.util.std_in_out.json_file_reader import JsonFileReader
from common.util.std_in_out.raw_file_reader import RawFileReader
from common.util.std_in_out.root_locator import RootLocator
from common.util.tagging.tagging_config_dto import TaggingConfigDTO
from framework.common.logger.message_type import MessageType
from logic_layer.rag_ingest.util.multi_stage_rag.chunk_generation.transformers.ktransformers_chunk_generator import (
    KTransformersChunkGenerator,
)


class TransformersTopicBase:
    def __init__(self, logger, tag_cfg=None):
        self.logger = logger
        self.tag_cfg = tag_cfg

        self.tokenizer = AutoTokenizer.from_pretrained(
            tag_cfg.tag_model if tag_cfg.tag_model is not None else
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.model = AutoModel.from_pretrained(tag_cfg.tag_model)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

        self.chunk_generator = KTransformersChunkGenerator(
            model_name=tag_cfg.tag_model if tag_cfg.tag_model is not None else
            "sentence-transformers/all-MiniLM-L6-v2",
            logger=self.logger
        )

        self.threshold = (
            TaggingConfigDTO.SIM_THRESHOLD_DEF
            if tag_cfg.sim_threshold is None
            else tag_cfg.sim_threshold
        )

    # ------------------------------------------------------
    def _encode(self, text: str):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True).to(self.device)
        with torch.no_grad():
            out = self.model(**inputs)
        emb = out.last_hidden_state[:, 0, :]
        return F.normalize(emb, p=2, dim=1)

    def _get_chunks(self, text, file_name,job_id=None):
        chunks = []
        try:
            if self.tag_cfg.is_K_Q_10_doc():
                extr = KQ10HtmlStructuredBlockExtractor()
                blocks = extr.extract_blocks(text)
                chunks = list(blocks.values())
            else:
                chunks.extend(self.chunk_generator.chunk(text,tag_dedup=self.tag_cfg.tag_dedup,job_id=job_id))
        except Exception as e:
            if self.logger:
                self.logger.do_log(
                    f"[TAGGING] ⚠ Failed chunking block in {file_name}: {e}",
                    2
                )
        return chunks

    def _extract_text(self, file_path: str, job_id: int):
        try:
            return RawFileReader.get_raw_text(file_path)
        except Exception as e:
            self._log_exception("[RANK] ❌ extract_text failed", e, job_id)
            return None

    def _extract_clean_text(self, file_path: str, job_id: int):
        try:

            warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

            raw = RawFileReader.get_raw_text(file_path)
            soup = BeautifulSoup(raw, "lxml")
            return soup.get_text(" ", strip=True)
        except Exception as e:
            self._log_exception("[RANK] ❌ extract_text failed", e, job_id)
            return None

    def _extract_chunks(self, text: str, file_name: str, job_id: int):
        try:
            chunks = self._get_chunks(text, file_name,job_id)
            if not chunks and self.logger:
                self.logger.do_log(
                    f"[RANK] ❌ No chunks generated | file={file_name}",
                    MessageType.INFO, job_id
                )
            return chunks
        except Exception as e:
            self._log_exception("[RANK] ❌ extract_chunks failed", e, job_id)
            return None

    def _log_exception(self, prefix: str, exc: Exception, job_id: int):
        tb = traceback.extract_tb(exc.__traceback__)
        last = tb[-1]
        msg = f"{prefix} | {exc.__class__.__name__}: {exc} | line={last.lineno} | file={last.filename}"
        if self.logger:
            self.logger.do_log(msg, MessageType.ERROR, job_id)

    def initialize_tag_dict(self, job_id):
        try:
            tag_dict = {}
            if self.tag_cfg.tag_file is not None:
                tag_dict = JsonFileReader.load_json_file(
                    os.path.join(RootLocator.get_root(), "static", "tags"),
                    self.tag_cfg.tag_file
                )
                self.logger.do_log(
                    f"[TAGGING] Loading tag file in tag folder static/tags  file={self.tag_cfg.tag_file}",
                    MessageType.INFO, job_id
                )
            elif self.tag_cfg.tag_json is not None:
                tag_dict = json.loads(self.tag_cfg.tag_json)
                self.logger.do_log(
                    f"[TAGGING] Loading tag file from tag_json field!  keys={tag_dict.keys()}",
                    MessageType.INFO, job_id
                )
            else:
                raise Exception("Missing tag_file or tag_json in the input file")
            return tag_dict
        except Exception as e:
            raise e
