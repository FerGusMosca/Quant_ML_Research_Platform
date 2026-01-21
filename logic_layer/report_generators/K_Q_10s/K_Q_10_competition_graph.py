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


class KQ10CompetitionGraph:
    """
    Builds a COMPETES_WITH graph from K-10 documents using
    semantic chunk scoring (no regex heuristics).

    Output edges are stored in-memory and are GraphRAG-ready.
    """

    # --- semantic intent seeds ---
    COMPETITION_PHRASES_HARD = [
        "competitors include",
        "principal competitors",
        "primary competitor",
        "significant competitors",
        "is a competitor"
    ]

    def __init__(self, logger, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        self.logger = logger
        self.edges: List[Dict] = []

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.nlp = spacy.load("en_core_web_sm")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

        self.hard_threshold = 0.5

        self.extractor = KQ10HtmlStructuredBlockExtractor()

        # Precompute individual embeddings for each HARD phrase (one-time)
        self.hard_phrase_embs = [self._encode(p) for p in self.COMPETITION_PHRASES_HARD]

        self.logger.do_log(
            "[KQ10CompetitionGraph] Initialized (semantic mode)",
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


    def _is_competitor_org(
            self,
            sec_symbol: str,
            org: str,
            hard_intent_emb,
    ) -> float:
        """
        Semantic validation using predefined TEMPLATES.
        Returns max similarity score across templates.
        """

        max_sim = 0.0

        for tpl in self.TEMPLATES:
            phrase = tpl.format(SEC=sec_symbol, ORG=org)
            emb = self._encode(phrase)
            sim = float(torch.dot(
                emb.squeeze(0),
                hard_intent_emb.squeeze(0)
            ))
            if sim > max_sim:
                max_sim = sim

        return max_sim

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
                f"[KQ10CompetitionGraph] ⚠ ORG extraction failed: {e}",
                MessageType.ERROR,
            )

        return list(orgs)

    def _extract_competition_sentences(
            self,
            texts: List[str]
    ) -> List[Dict[str, any]]:
        """
        Return list of dicts with competition-related sentences and their similarity scores.
        """
        competition_items = []

        for doc in self.nlp.pipe(texts):
            for sent in doc.sents:
                sent_text = sent.text.strip()

                if len(sent_text) < 30:  # optional minimal length filter
                    continue

                emb = self._encode(sent_text)
                sims = [float(torch.dot(emb.squeeze(0), p_emb.squeeze(0))) for p_emb in self.hard_phrase_embs]
                sim = max(sims) if sims else 0.0

                if sim >= self.hard_threshold:
                    competition_items.append({
                        "sentence": sent_text,
                        "score": round(sim, 4)
                    })
                    self.logger.do_log(
                        f"[CompetitionGraph] Added sentence (score {round(sim, 4)}): {sent_text[:80]}...",
                        MessageType.INFO
                    )

        return competition_items

    # ------------------------------------------------------------------
    def extract_competition(self, matched_file: SecurityWithFile, job_id: int = None):
        sec_symbol = matched_file.security.symbol
        file_path = matched_file.file
        file_name = os.path.basename(file_path)

        self.logger.do_log(
            f"[KQ10CompetitionGraph] ▶ Processing K10 | security={sec_symbol} | file={file_name}",
            MessageType.INFO,
            job_id,
        )

        # --------------------------------------------------
        # Read file
        # --------------------------------------------------
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                raw_text = f.read()
                #raw_text = self._extract_item1_business(raw_text)
        except Exception as e:
            self._log_exception("[KQ10CompetitionGraph] ❌ file read failed", e, job_id)
            return

        # --------------------------------------------------
        # Extract structured K-10 blocks_lines
        # --------------------------------------------------
        try:
            blocks_lines , blocks_text= self.extractor.extract_blocks_adv(raw_text, ["COMPETITION"])
        except Exception as e:
            self._log_exception("[KQ10CompetitionGraph] ❌ block extraction failed", e, job_id)
            return

        if not blocks_lines:
            return

        # --------------------------------------------------
        # Aggregate Item 1 blocks_lines
        # --------------------------------------------------
        '''
        item1_texts = [
            block_text
            for block_id, block_text in blocks_lines.items()
            if "item 1" in block_id.lower()
        ]
        '''
        item1_texts = list(blocks_lines["COMPETITION"])

        # --------------------------------------------------
        # Sentence-level competition detection
        # --------------------------------------------------
        competition_sentences = self._extract_competition_sentences(
            item1_texts
        )

        sorted_competition = sorted(
            competition_sentences,
            key=lambda x: x["score"],
            reverse=True
        )

        for sentence in sorted_competition:
            orgs = self._extract_org_entities(sentence["sentence"])
            if not orgs:
                continue

            for org in orgs:
                edge = {
                    "src": sec_symbol,
                    "relation": "COMPETES_WITH",
                    "dst": org,
                    "file": file_name,
                    "block_id": "ITEM_1_SENTENCE",
                    "score": round(sentence["score"], 4),
                }
                self.edges.append(edge)
                self.logger.do_log(
                    f"[CompetitionGraph] Added edge: {sec_symbol} -> {org} (score {round(sentence['score'], 4)})",
                    MessageType.INFO
                )

        # --------------------------------------------------
        # Optional: sort for inspection (no truncation here)
        # --------------------------------------------------
        self.edges.sort(key=lambda x: x["score"], reverse=True)

        self.logger.do_log(
            f"[KQ10CompetitionGraph] ✔ Competition sentences processed | "
            f"security={sec_symbol} | edges={len(self.edges)}",
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
