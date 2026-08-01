# FILE: bert_topic_tagger.py
import json
import os.path
import csv
import traceback
from datetime import datetime
import os
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from common.util.extractors.K_Q_10.k_q_10_html_structured_block_extractor import KQ10HtmlStructuredBlockExtractor
from common.util.std_in_out.json_file_reader import JsonFileReader
from common.util.std_in_out.raw_file_reader import RawFileReader
from common.util.std_in_out.root_locator import RootLocator
from common.util.tagging.tagging_config_dto import TaggingConfigDTO
from framework.common.logger.message_type import MessageType
from logic_layer.rag_corpus_metadata.financial_tags import FINANCIAL_TAGS
from logic_layer.rag_corpus_metadata.tagger.transformers_topic_base import TransformersTopicBase
from logic_layer.rag_ingest.util.legacy_rag.pdf_cleaner import PDFCleaner
from logic_layer.rag_ingest.util.multi_stage_rag.chunk_generation.transformers.ktransformers_chunk_generator import \
    KTransformersChunkGenerator


class TransformersTopicTagger(TransformersTopicBase):

    LOG_SIM_FLOOR = 0.60   # only log individual chunks above this similarity

    def __init__(self, logger, tag_cfg=None):
        super().__init__(logger, tag_cfg)

        if tag_cfg.tag_file is None:
            if tag_cfg.tags_csv is not None:
                self.keywords = tag_cfg.tags_csv
            else:
                self.keywords = FINANCIAL_TAGS
        else:
            root_path = RootLocator.get_root()
            tag_file = os.path.join(root_path, "static", "tags", tag_cfg.tag_file)
            with open(tag_file, "r", encoding="utf-8") as f:
                self.keywords = json.load(f)

        self.logger.do_log("[BERT] Topic tagger READY", 1)

    def _persist_rank_csv(self, rows: list, output_dir: str, job_id: int = None):

        if not rows:
            self.logger.do_log("[RANK][CSV] ⚠ No rows to persist", MessageType.INFO, job_id)
            return

        output_dir = os.path.abspath(os.path.expanduser(str(output_dir).strip()))
        os.makedirs(output_dir, exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        out_file = os.path.join(output_dir, f"rank.csv")

        fieldnames = ["security", "file", "rank_1", "rank_2", "rank_3"]

        try:
            with open(out_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

            self.logger.do_log(
                f"[RANK][CSV] ✔ Persisted results | file={out_file} | rows={len(rows)}",
                MessageType.INFO,
                job_id
            )

        except Exception as e:
            self._log_exception("[RANK] ❌ _persist_rank_csv failed", e, job_id)

    def _score_chunks(
            self,
            chunks: list,
            tag_embeddings: dict,
            file_name: str,
            security_symbol: str,
            job_id: int
    ):
        """Scores every chunk against every tag phrase, keeping the best match per chunk."""
        scores = []
        best = {"sim": 0.0, "tag": None, "phrase": None, "chunk": None, "idx": None}

        for chunk_idx, chunk in enumerate(chunks, start=1):
            try:
                chunk_emb = self._encode(chunk).squeeze(0)
                max_sim, hit_tag, hit_phrase = 0.0, None, None

                for tag, emb_list in tag_embeddings.items():
                    for phrase_idx, we in enumerate(emb_list):
                        sim = float(torch.dot(chunk_emb, we.squeeze(0)))
                        if sim > max_sim:
                            max_sim, hit_tag, hit_phrase = sim, tag, phrase_idx

                scores.append(max_sim)

                if max_sim > best["sim"]:
                    best = {"sim": max_sim, "tag": hit_tag, "phrase": hit_phrase,
                            "chunk": chunk, "idx": chunk_idx}

                # per-chunk detail only for meaningful hits
                if self.logger and max_sim >= self.LOG_SIM_FLOOR:
                    self.logger.do_log(
                        f"[RANK] {security_symbol} | chunk {chunk_idx}/{len(chunks)} | "
                        f"sim={max_sim:.4f} | tag={hit_tag}#{hit_phrase} | {chunk[:120]!r}",
                        MessageType.INFO, job_id
                    )

            except Exception as e:
                self._log_exception("[RANK] ❌ scoring failed", e, job_id)

        if self.logger and scores:
            avg = sum(scores) / len(scores)
            top = sorted(scores, reverse=True)[:3]
            self.logger.do_log(
                f"[RANK] ✔ {security_symbol} | file={file_name} | chunks={len(chunks)} | "
                f"avg={avg:.4f} | top3={[round(t, 4) for t in top]}",
                MessageType.INFO, job_id
            )
            if best["chunk"]:
                self.logger.do_log(
                    f"[RANK] 🏆 BEST {security_symbol} | sim={best['sim']:.4f} | "
                    f"tag={best['tag']}#{best['phrase']} | chunk {best['idx']}/{len(chunks)}\n"
                    f"        {best['chunk'][:400]}",
                    MessageType.INFO, job_id
                )

        return scores

    def _build_ranking(self, security_tag_scores: dict, top_k: int, job_id: int):
        results = []

        for security, scores in security_tag_scores.items():
            try:
                ranked = sorted(scores, reverse=True)

                if self.logger:
                    self.logger.do_log(
                        f"[RANK] security={security} | ranked={ranked[:top_k]}",
                        MessageType.INFO, job_id
                    )

                results.append({
                    "security": security,
                    "rank_1": ranked[0] if len(ranked) > 0 else 0,
                    "rank_2": ranked[1] if len(ranked) > 1 else 0,
                    "rank_3": ranked[2] if len(ranked) > 2 else 0,
                })

            except Exception as e:
                self._log_exception("[RANK] ❌ ranking build failed", e, job_id)

        results.sort(key=lambda x: x["rank_1"], reverse=True)
        return results

    # ------------------------------------------------------

    def classify(self, text: str, file_name: str):
        """
        Semantic topic tagging using chunk-level BERT similarity.
        Blocks are treated as the primary structural units.
        """

        # Convert structured blocks into chunkable texts

        # Generate chunks per block (do NOT chunk the whole document at once)
        chunks = self._get_chunks(text,file_name)
        if not chunks:
            if self.logger:
                self.logger.do_log(
                    f"[TAGGING] ❌ No chunks generated for {file_name}",
                    2
                )
            return ["uncertain"]

        # Precompute embeddings for chunks
        chunk_embeddings = [self._encode(c) for c in chunks]

        tags = set()

        # Evaluate each topic against all chunks
        for topic, words in self.keywords.items():
            topic_embs = [self._encode(w) for w in words]
            max_sim = 0.0

            for ce in chunk_embeddings:
                for we in topic_embs:
                    ce = ce.squeeze(0)
                    we = we.squeeze(0)
                    sim = float(torch.dot(ce, we))
                    if sim > max_sim:
                        max_sim = sim

            if max_sim > self.threshold:
                tags.add(topic)

            if self.logger:
                self.logger.do_log(
                    f"[TAGGING] file={file_name} | topic={topic} | threshold={self.threshold} | max_sim={max_sim}",
                    2
                )

        if not tags:
            tags = {"uncertain"}

        return list(tags)

    def rank(
            self,
            securities: list,
            sec_w_files: list,
            rank_folder,
            tag_dict: dict,
            job_id: int,
            top_k: int = 3,
    ):
        if self.logger:
            self.logger.do_log(
                f"[RANK] ▶ START | files={len(sec_w_files)} | tags={len(tag_dict)} | top_k={top_k}",
                MessageType.INFO, job_id
            )

        security_tag_scores = {sec.symbol: [] for sec in securities}

        tag_embeddings = {
            tag: [self._encode(p) for p in phrases]
            for tag, phrases in tag_dict.items()
        }

        for idx, sec_w_file in enumerate(sec_w_files, start=1):
            file_path=sec_w_file.file
            file_name = os.path.basename(file_path)
            security_symbol=sec_w_file.security.symbol

            if self.logger:
                self.logger.do_log(
                    f"[RANK] ▶ ({idx}/{len(sec_w_files)}) file={file_name}",
                    MessageType.INFO, job_id
                )
            '''
            security_symbol = next(
                (sec.symbol for sec in securities if file_name.startswith(sec.symbol+"_")),
                None
            )
            '''

            if not security_symbol:
                continue

            text = self._extract_text(file_path, job_id)
            if not text:
                continue

            chunks = self._extract_chunks(text, file_name, job_id)
            if not chunks:
                continue

            scores = self._score_chunks(
                chunks,
                tag_embeddings,
                file_name,
                security_symbol,
                job_id
            )

            security_tag_scores[security_symbol].extend(scores)

        results = self._build_ranking(security_tag_scores, top_k, job_id)

        if self.logger:
            self.logger.do_log(
                f"[RANK] ✔ DONE | securities={len(results)}",
                MessageType.INFO, job_id
            )

        self._persist_rank_csv(results, rank_folder, job_id)
        return results