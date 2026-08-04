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

    MIN_BLOCK_CHARS = 200   # skip 'None.' style empty items

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

        self.pooling = self._resolve_pooling()

        self.threshold = (
            TaggingConfigDTO.SIM_THRESHOLD_DEF
            if tag_cfg.sim_threshold is None
            else tag_cfg.sim_threshold
        )

    # ------------------------------------------------------
    # ------------------------------------------------------
    # Pooling strategy per model family.
    # Using the wrong one silently flattens cosine similarities.
    POOLING_BY_MODEL = {
        "BAAI/bge-small-en-v1.5": "cls",          # BGE models are trained on the CLS token
        "BAAI/bge-base-en-v1.5": "cls",
        "BAAI/bge-large-en-v1.5": "cls",
        "sentence-transformers/all-MiniLM-L6-v2": "mean",
        "sentence-transformers/all-mpnet-base-v2": "mean",
        "distilbert-base-uncased": "mean",
    }
    DEFAULT_POOLING = "mean"
    ENCODE_BATCH_SIZE = 32     # texts per forward pass; raise it if you have spare RAM/VRAM

    def _resolve_pooling(self):
        """Returns 'cls' or 'mean' for the configured model, warning on unknown ones."""
        model_name = self.tag_cfg.tag_model if self.tag_cfg.tag_model is not None \
            else "sentence-transformers/all-MiniLM-L6-v2"

        pooling = self.POOLING_BY_MODEL.get(model_name)

        if pooling is None:
            # loose match on the vendor prefix before giving up (e.g. other BAAI/bge variants)
            for known, strategy in self.POOLING_BY_MODEL.items():
                if model_name.lower().startswith(known.split("/")[0].lower()):
                    pooling = strategy
                    break

        if pooling is None:
            pooling = self.DEFAULT_POOLING
            if self.logger:
                self.logger.do_log(
                    f"[ENCODE] Unknown model '{model_name}' - falling back to "
                    f"'{self.DEFAULT_POOLING}' pooling. Check its model card and add it to "
                    f"POOLING_BY_MODEL if it expects CLS.",
                    MessageType.INFO
                )
        elif self.logger:
            self.logger.do_log(
                f"[ENCODE] model={model_name} | pooling={pooling}",
                MessageType.INFO
            )

        return pooling

    def _encode(self, text: str):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True).to(self.device)
        with torch.no_grad():
            out = self.model(**inputs)

        if self.pooling == "cls":
            emb = out.last_hidden_state[:, 0, :]
        else:
            # mean pooling over real tokens only (padding excluded via the attention mask)
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            emb = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

        return F.normalize(emb, p=2, dim=1)

    def _encode_batch(self, texts: list, batch_size: int = None):
        """
        Encodes a list of texts in batches, returning a normalized (n_texts, dim) tensor.
        One forward pass per batch instead of one per text - this is where the speedup is.
        """
        if not texts:
            return torch.empty(0)

        batch_size = batch_size or self.ENCODE_BATCH_SIZE
        out_chunks = []

        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                padding=True,          # required to stack texts of different lengths
                max_length=512,
            ).to(self.device)

            with torch.no_grad():
                out = self.model(**inputs)

            if self.pooling == "cls":
                emb = out.last_hidden_state[:, 0, :]
            else:
                mask = inputs["attention_mask"].unsqueeze(-1).float()
                emb = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

            out_chunks.append(F.normalize(emb, p=2, dim=1))

        return torch.cat(out_chunks, dim=0)

    def _get_chunks(self, text, file_name, job_id=None):
        chunks = []
        try:
            if self.tag_cfg.is_K_Q_10_doc():
                # 1) structural split: one block per 10-K/10-Q Item
                extr = KQ10HtmlStructuredBlockExtractor()
                blocks = extr.extract_blocks(text)

                # 2) semantic split: each block is broken down into ~180-word chunks
                for item, block in blocks.items():
                    if not block or len(block) < self.MIN_BLOCK_CHARS:
                        continue
                    try:
                        sub = self.chunk_generator.chunk(
                            block, tag_dedup=self.tag_cfg.tag_dedup, job_id=job_id
                        )
                        chunks.extend(sub)
                        if self.logger:
                            self.logger.do_log(
                                f"[TAGGING] block={item} | chars={len(block)} | chunks={len(sub)}",
                                MessageType.INFO, job_id
                            )
                    except Exception as be:
                        if self.logger:
                            self.logger.do_log(
                                f"[TAGGING] block={item} chunking failed: {be}",
                                MessageType.INFO, job_id
                            )
            else:
                chunks.extend(self.chunk_generator.chunk(text, tag_dedup=self.tag_cfg.tag_dedup, job_id=job_id))

            if self.logger:
                self.logger.do_log(
                    f"[TAGGING] 📄 {file_name} | blocks -> {len(chunks)} chunks",
                    MessageType.INFO, job_id
                )

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
