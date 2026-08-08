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
from logic_layer.rag_corpus_metadata.tagger.chunk_sources.chunk_source_factory import ChunkSourceFactory
from logic_layer.rag_corpus_metadata.tagger.transformers_topic_base import TransformersTopicBase
from logic_layer.rag_ingest.util.legacy_rag.pdf_cleaner import PDFCleaner
from logic_layer.rag_ingest.util.multi_stage_rag.chunk_generation.transformers.ktransformers_chunk_generator import \
    KTransformersChunkGenerator


class TransformersTopicTagger(TransformersTopicBase):

    LOG_SIM_FLOOR = 0.60      # only log individual chunks above this similarity
    DENSITY_FLOOR = 0.65      # a chunk counts as a "hit" above this cosine similarity
    DENSITY_WEIGHT = 0.5      # weight of density vs peak score in the composite ranking

    def __init__(self, logger, tag_cfg=None, vectors_db_config=None):
        super().__init__(logger, tag_cfg)

        # Only used when the run reads its chunks from pgvector
        self.vectors_db_config = vectors_db_config

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

        # density floor can be overridden per run via the tagging config
        cfg_floor = getattr(tag_cfg, "density_floor", None)
        self.density_floor = cfg_floor if cfg_floor is not None else self.DENSITY_FLOOR

        self.logger.do_log(
            f"[BERT] Topic tagger READY | density_floor={self.density_floor} | "
            f"chunk_source={tag_cfg.chunk_source}", 1
        )

    def _persist_rank_csv(self, rows: list, output_dir: str, job_id: int = None):

        if not rows:
            self.logger.do_log("[RANK][CSV] ⚠ No rows to persist", MessageType.INFO, job_id)
            return

        output_dir = os.path.abspath(os.path.expanduser(str(output_dir).strip()))
        os.makedirs(output_dir, exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        out_file = os.path.join(output_dir, f"rank.csv")

        fieldnames = ["security", "file", "rank_1", "rank_2", "rank_3",
                      "density", "hits", "chunk_count", "coverage", "composite"]

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
            tag_matrix,
            tag_index: list,
            file_name: str,
            security_symbol: str,
            job_id: int,
            chunk_matrix=None,
            mode: str = None
    ):
        """
        Encodes all chunks in batches and scores them against the tag matrix in a single
        matmul, giving a (n_chunks, n_phrases) similarity matrix.
        Returns (scores, hit_phrases): best similarity per chunk, plus every tag phrase
        that fired above the density floor (used for coverage).
        """
        scores = []
        hit_phrases = set()

        try:
            start_time = datetime.now()

            # (n_chunks, dim). In VECTORS mode the chunk source already carried
            # the embeddings, so this is free; in FILES mode it is the expensive
            # part of the whole run.
            if chunk_matrix is None:
                chunk_matrix = self._encode_batch(chunks)

            # (n_chunks, n_phrases) - every chunk against every phrase at once
            sims = torch.matmul(chunk_matrix, tag_matrix.T)

            best_per_chunk, best_pos = torch.max(sims, dim=1)
            scores = [float(v) for v in best_per_chunk]

            for pos in torch.nonzero(sims >= self.density_floor)[:, 1].unique().tolist():
                hit_phrases.add(tag_index[pos])

            elapsed = (datetime.now() - start_time).total_seconds()

            if self.logger:
                for chunk_idx in torch.nonzero(best_per_chunk >= self.LOG_SIM_FLOOR).flatten().tolist():
                    self.logger.do_log(
                        f"[RANK][{mode}] {security_symbol} | chunk {chunk_idx + 1}/{len(chunks)} | "
                        f"sim={scores[chunk_idx]:.4f} | tag={tag_index[int(best_pos[chunk_idx])]} | "
                        f"{chunks[chunk_idx][:120]!r}",
                        MessageType.INFO, job_id
                    )

                hits = sum(1 for v in scores if v >= self.density_floor)
                self.logger.do_log(
                    f"[RANK][{mode}] ✔ {security_symbol} | file={file_name} | chunks={len(chunks)} | "
                    f"hits={hits} | density={hits / len(scores):.4f} | coverage={len(hit_phrases)} | "
                    f"top3={[round(t, 4) for t in sorted(scores, reverse=True)[:3]]} | "
                    f"{elapsed:.1f}s ({len(chunks) / max(elapsed, 0.001):.1f} chunks/s)",
                    MessageType.INFO, job_id
                )

                top_idx = int(torch.argmax(best_per_chunk))
                self.logger.do_log(
                    f"[RANK][{mode}] 🏆 BEST {security_symbol} | sim={scores[top_idx]:.4f} | "
                    f"tag={tag_index[int(best_pos[top_idx])]} | chunk {top_idx + 1}/{len(chunks)}\n"
                    f"        {chunks[top_idx][:400]}",
                    MessageType.INFO, job_id
                )

        except Exception as e:
            self._log_exception("[RANK] ❌ scoring failed", e, job_id)

        return scores, hit_phrases

    def _build_ranking(self, security_tag_scores: dict, security_hits: dict,
                       top_k: int, job_id: int):
        """
        Builds one row per security combining two independent signals:
          - peak    (rank_1/2/3): the strongest single statement in the filing
          - density: share of chunks above the floor, i.e. how pervasive the topic is
          - coverage: how many DIFFERENT tag phrases fired (a topic discussed from
                      several angles beats the same sentence repeated)
        """
        results = []

        for security, scores in security_tag_scores.items():
            try:
                ranked = sorted(scores, reverse=True)
                total = len(ranked)
                hits = sum(1 for s in ranked if s >= self.density_floor)
                density = hits / total if total else 0.0
                coverage = len(security_hits.get(security, set()))
                peak = ranked[0] if total else 0.0

                # density is bounded but tiny in absolute terms, so it is amplified
                # before mixing: a 10% hit rate is already a very topical document
                composite = ((1 - self.DENSITY_WEIGHT) * peak +
                             self.DENSITY_WEIGHT * min(density * 10, 1.0))

                results.append({
                    "security": security,
                    "rank_1": peak,
                    "rank_2": ranked[1] if total > 1 else 0,
                    "rank_3": ranked[2] if total > 2 else 0,
                    "density": round(density, 6),
                    "hits": hits,
                    "chunk_count": total,
                    "coverage": coverage,
                    "composite": round(composite, 6),
                })

            except Exception as e:
                self._log_exception("[RANK] ❌ ranking build failed", e, job_id)

        results.sort(key=lambda x: x["composite"], reverse=True)
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
            fiscal_year=None,
            quarter=None,
    ):
        # Where the chunks come from is decided once, up front, and printed loud:
        # a run scored against stale vectors looks exactly like a run scored
        # against the files unless the mode is visible.
        chunk_source = ChunkSourceFactory.build(
            self.tag_cfg.chunk_source, self, self.logger, self.vectors_db_config
        )
        mode = chunk_source.MODE

        if self.logger:
            self.logger.do_log(
                f"[RANK][{mode}] ▶ START | files={len(sec_w_files)} | tags={len(tag_dict)} | "
                f"top_k={top_k} | model={self.tag_cfg.tag_model} | source={chunk_source.DESCRIPTION}",
                MessageType.INFO, job_id
            )

        security_tag_scores = {sec.symbol: [] for sec in securities}
        security_hits = {sec.symbol: set() for sec in securities}

        # flatten every tag phrase into a single (n_phrases, dim) matrix so each chunk
        # is scored with one matmul instead of a nested python loop
        tag_index = []
        tag_phrases = []
        for tag, phrases in tag_dict.items():
            for phrase_idx, phrase in enumerate(phrases):
                tag_index.append(f"{tag}#{phrase_idx}")
                tag_phrases.append(phrase)
        tag_matrix = self._encode_batch(tag_phrases)

        if self.logger:
            self.logger.do_log(
                f"[RANK] tag matrix ready | phrases={len(tag_index)} | dim={tag_matrix.shape[1]}",
                MessageType.INFO, job_id
            )

        files_scored = 0
        files_without_chunks = 0

        try:
            for idx, sec_w_file in enumerate(sec_w_files, start=1):
                file_path = sec_w_file.file
                file_name = os.path.basename(file_path)
                security_symbol = sec_w_file.security.symbol

                if self.logger:
                    self.logger.do_log(
                        f"[RANK][{mode}] ▶ ({idx}/{len(sec_w_files)}) file={file_name}",
                        MessageType.INFO, job_id
                    )

                if not security_symbol:
                    continue

                # chunk_matrix comes back filled in VECTORS mode and None in
                # FILES mode, which is what decides whether we encode
                chunks, chunk_matrix = chunk_source.get_chunks(
                    sec_w_file, file_name, fiscal_year, quarter, job_id
                )

                if not chunks:
                    files_without_chunks += 1
                    continue

                scores, hit_phrases = self._score_chunks(
                    chunks,
                    tag_matrix,
                    tag_index,
                    file_name,
                    security_symbol,
                    job_id,
                    chunk_matrix=chunk_matrix,
                    mode=mode
                )

                files_scored += 1
                security_tag_scores[security_symbol].extend(scores)
                security_hits[security_symbol].update(hit_phrases)

        finally:
            chunk_source.close()

        results = self._build_ranking(security_tag_scores, security_hits, top_k, job_id)

        if self.logger:
            self.logger.do_log(
                f"[RANK][{mode}] ✔ DONE | securities={len(results)} | "
                f"files_scored={files_scored}/{len(sec_w_files)} | "
                f"files_without_chunks={files_without_chunks}",
                MessageType.INFO, job_id
            )

        self._persist_rank_csv(results, rank_folder, job_id)
        return results





