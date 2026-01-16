from typing import List, Dict
import os
import traceback

import spacy
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

from common.dto.sec_w_file import SecurityWithFile
from common.util.extractors.K_Q_10.k_q_10_html_structured_block_extractor import (
    KQ10HtmlStructuredBlockExtractor
)
from framework.common.logger.message_type import MessageType


class CompetitionGraph:
    """
    Builds a COMPETES_WITH graph from K-10 documents using
    semantic chunk scoring (no regex heuristics).

    Output edges are stored in-memory and are GraphRAG-ready.
    """

    # --- semantic intent seeds ---
    COMPETITION_PHRASES_HARD = [
        "we compete with",
        "our principal competitors include",
        "we face direct competition from",
        "our main competitors are",
    ]

    COMPETITION_PHRASES_SOFT = [
        "competitive landscape",
        "intense competition",
        "highly competitive market",
        "competitive pressures",
    ]

    EXCLUSION_HINTS = [
        "customer",
        "supplier",
        "partner",
        "subsidiary",
        "affiliate",
        "regulator",
        "government",
    ]

    def __init__(self, logger, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.logger = logger
        self.edges: List[Dict] = []

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.nlp = spacy.load("en_core_web_sm")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

        self.hard_threshold = 0.72
        self.soft_threshold = 0.60
        self.extractor = KQ10HtmlStructuredBlockExtractor()

        self.logger.do_log(
            "[CompetitionGraph] Initialized (semantic mode)",
            MessageType.INFO,
        )

    # ------------------------------------------------------------------
    # Embedding utils
    # ------------------------------------------------------------------
    def _encode(self, text: str):
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True
        ).to(self.device)
        with torch.no_grad():
            out = self.model(**inputs)
        emb = out.last_hidden_state[:, 0, :]
        return F.normalize(emb, p=2, dim=1)

    def _build_competition_intent_embeddings(self):
        hard_embs = [self._encode(p) for p in self.COMPETITION_PHRASES_HARD]
        soft_embs = [self._encode(p) for p in self.COMPETITION_PHRASES_SOFT]

        hard_intent = torch.mean(torch.stack(hard_embs), dim=0)
        soft_intent = torch.mean(torch.stack(soft_embs), dim=0)

        return hard_intent, soft_intent

    # ------------------------------------------------------------------
    # VERY DELIBERATE placeholder
    # ------------------------------------------------------------------
    def _extract_org_entities(self, text: str) -> List[str]:
        """
        Extract organization entities from competition-related blocks only.
        Returns raw organization names (no symbol resolution here).
        """

        orgs = set()

        try:
            doc = self.nlp(text)
            for ent in doc.ents:
                if ent.label_ == "ORG":
                    val = ent.text.strip()
                    if len(val) >= 3:
                        orgs.add(val)
        except Exception as e:
            self.logger.do_log(
                f"[CompetitionGraph] ⚠ ORG extraction failed: {e}",
                MessageType.ERROR,
            )

        return list(orgs)

    def _extract_item1_business(self, text: str) -> str:
        t = text.lower()

        start = t.find("item 1.")
        if start == -1:
            start = t.find("item 1 ")

        if start == -1:
            return text

        end_candidates = [
            t.find("item 1a.", start),
            t.find("item 1a ", start),
            t.find("item 2.", start),
            t.find("item 2 ", start),
        ]
        end_candidates = [e for e in end_candidates if e != -1]

        end = len(text)
        return text[start:end]

    # ------------------------------------------------------------------
    def extract_competition(self, matched_file: SecurityWithFile, job_id: int = None):
        sec_symbol = matched_file.security.symbol
        file_path = matched_file.file
        file_name = os.path.basename(file_path)

        self.logger.do_log(
            f"[CompetitionGraph] ▶ Processing K10 | security={sec_symbol} | file={file_name}",
            MessageType.INFO,
            job_id,
        )

        # --------------------------------------------------
        # Read file
        # --------------------------------------------------
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                raw_text = f.read()
                raw_text = self._extract_item1_business(raw_text)
        except Exception as e:
            self._log_exception("[CompetitionGraph] ❌ file read failed", e, job_id)
            return

        # --------------------------------------------------
        # Extract structured K-10 blocks
        # --------------------------------------------------
        try:
            blocks = self.extractor.extract_blocks(raw_text)
        except Exception as e:
            self._log_exception("[CompetitionGraph] ❌ block extraction failed", e, job_id)
            return

        if not blocks:
            self.logger.do_log(
                f"[CompetitionGraph] ⚠ No blocks extracted | file={file_name}",
                MessageType.INFO,
                job_id,
            )
            return

        # --------------------------------------------------
        # Build intent embeddings
        # --------------------------------------------------
        hard_intent_emb, soft_intent_emb = self._build_competition_intent_embeddings()

        # --------------------------------------------------
        # Process blocks
        # --------------------------------------------------
        for block_id, block_text in blocks.items():
            try:
                block_id_l = block_id.lower()

                if not (
                        "item 1" in block_id_l
                        or "business" in block_id_l
                        or "competition" in block_id_l
                ):
                    continue

                chunk_emb = self._encode(block_text)

                sim_hard = float(
                    torch.dot(chunk_emb.squeeze(0), hard_intent_emb.squeeze(0))
                )
                sim_soft = float(
                    torch.dot(chunk_emb.squeeze(0), soft_intent_emb.squeeze(0))
                )

                if sim_hard < self.hard_threshold:
                    continue

                text_l = block_text.lower()
                if not any(p in text_l for p in self.COMPETITION_PHRASES_HARD):
                    continue

                orgs = self._extract_org_entities(block_text)
                if not orgs:
                    continue

                block_l = block_text.lower()

                for org in orgs:
                    # Final role cleanup (customers, suppliers, etc.)
                    if any(h in block_l for h in self.EXCLUSION_HINTS):
                        continue

                    edge = {
                        "src": sec_symbol,
                        "relation": "COMPETES_WITH",
                        "dst": org,
                        "file": file_name,
                        "block_id": block_id,
                        "score": round(sim_hard, 4),
                        "soft_score": round(sim_soft, 4),
                    }

                    self.edges.append(edge)

                self.logger.do_log(
                    f"[CompetitionGraph] ✔ competition block | security={sec_symbol} "
                    f"| block={block_id} | hard={sim_hard:.3f} | soft={sim_soft:.3f}",
                    MessageType.INFO,
                    job_id,
                )

            except Exception as e:
                self._log_exception(
                    "[CompetitionGraph] ❌ block processing failed",
                    e,
                    job_id,
                )

        self.logger.do_log(
            f"[CompetitionGraph] ✔ DONE | security={sec_symbol} | edges={len(self.edges)}",
            MessageType.INFO,
            job_id,
        )

    # ------------------------------------------------------------------
    def get_edges(self) -> List[Dict]:
        """
        Final materialization point.
        This output is suitable for:
        - Neo4j ingestion
        - GraphRAG indexing
        - Edge weighting / centrality
        """
        return self.edges

    # ------------------------------------------------------------------
    def _log_exception(self, prefix: str, exc: Exception, job_id: int):
        tb = traceback.extract_tb(exc.__traceback__)
        last = tb[-1]
        msg = (
            f"{prefix} | {exc.__class__.__name__}: {exc} "
            f"| line={last.lineno} | file={last.filename}"
        )
        self.logger.do_log(msg, MessageType.ERROR, job_id)
