# FILE: bert_topic_tagger.py
import json
import os.path
import csv
from datetime import datetime
import os
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from common.util.extractors.K_Q_10.k_q_10_html_structured_block_extractor import KQ10HtmlStructuredBlockExtractor
from common.util.std_in_out.raw_file_reader import RawFileReader
from common.util.std_in_out.root_locator import RootLocator
from common.util.tagging.tagging_config_dto import TaggingConfigDTO
from framework.common.logger.message_type import MessageType
from logic_layer.rag_corpus_metadata.financial_tags import FINANCIAL_TAGS
from logic_layer.rag_ingest.util.legacy_rag.pdf_cleaner import PDFCleaner
from logic_layer.rag_ingest.util.multi_stage_rag.chunk_generation.transformers.ktransformers_chunk_generator import \
    KTransformersChunkGenerator


class TransformersTopicTagger:
    def __init__(self, logger, tag_cfg=None):
        self.logger = logger
        self.tokenizer = AutoTokenizer.from_pretrained(tag_cfg.tag_model if tag_cfg.tag_model is not None else "sentence-transformers/all-MiniLM-L6-v2")
        self.model = AutoModel.from_pretrained(tag_cfg.tag_model)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.tag_cfg=tag_cfg

        self.chunk_generator = KTransformersChunkGenerator(model_name=tag_cfg.tag_model if tag_cfg.tag_model is not None else "sentence-transformers/all-MiniLM-L6-v2", logger=self.logger)

        self.threshold=TaggingConfigDTO.SIM_THRESHOLD_DEF if tag_cfg.sim_threshold is None else tag_cfg.sim_threshold

        if tag_cfg.tag_file is None:
            if tag_cfg.tags_csv is not None:
                self.keywords=tag_cfg.tags_csv
            else:
                self.keywords = FINANCIAL_TAGS
        else:
            root_path=RootLocator.get_root()
            tag_file=os.path.join(root_path,"static","tags",tag_cfg.tag_file)
            with open(tag_file, "r", encoding="utf-8") as f:
                self.keywords = json.load(f)

        self.logger.do_log("[BERT] Topic tagger READY", 1)

    # ------------------------------------------------------
    def _encode(self, text: str):
        """Returns Transfomers sentence embedding."""
        inputs = self.tokenizer(text, return_tensors="pt",
                                truncation=True).to(self.device)
        with torch.no_grad():
            out = self.model(**inputs)
        emb = out.last_hidden_state[:, 0, :]
        return F.normalize(emb, p=2, dim=1)


    def _get_chunks(self,text,file_name):
        chunks = []
        try:
            if self.tag_cfg.is_K_Q_10_doc():
                extr=KQ10HtmlStructuredBlockExtractor()
                blocks=extr.extract_blocks(text)
                chunks = list(blocks.values())
            else:#Generic Tag Manager for Plain Text
                chunks.extend(self.chunk_generator.chunk(text))
        except Exception as e:
            if self.logger:
                self.logger.do_log(
                    f"[TAGGING] ⚠ Failed chunking block in {file_name}: {e}",
                    2
                )

        return chunks

    def _persist_rank_csv(self, rows: list, output_dir: str,job_id:int=None):


        if not rows:
            if self.logger:
                self.logger.do_log("[RANK][CSV] ⚠ No rows to persist", MessageType.INFO,job_id)
            return

        os.makedirs(output_dir, exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        out_file = os.path.join(output_dir, f"rank_results_{ts}.csv")

        fieldnames = ["security", "file", "rank_1", "rank_2", "rank_3"]

        try:
            with open(out_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for r in rows:
                    writer.writerow(r)

            if self.logger:
                self.logger.do_log(
                    f"[RANK][CSV] ✔ Persisted results | file={out_file} | rows={len(rows)}",
                    MessageType.INFO,job_id
                )

        except Exception as e:
            if self.logger:
                self.logger.do_log(
                    f"[RANK][CSV] ❌ Failed persisting CSV: {e}",
                    MessageType.ERROR,job_id
                )


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
            files: list,
            rank_folder,
            tag_dict: dict,
            job_id:int,
            top_k: int = 3,
    ):
        """
        ONE ROW PER SECURITY.
        ALWAYS produces rank_1, rank_2, rank_3 (no threshold).
        Ordered by rank_1 score DESC.
        """
        # security -> tag -> max_score
        security_tag_scores = {}

        if self.logger:
            self.logger.do_log(
                f"[RANK] ▶ START | files={len(files)} | tags={len(tag_dict)} | top_k={top_k}",
                MessageType.INFO,job_id
            )

        # Pre-encode tag phrases once
        tag_embeddings = {}

        for sec in securities:
            security_tag_scores[sec.symbol] = []

        for tag, phrases in tag_dict.items():
            tag_embeddings[tag] = [self._encode(p) for p in phrases]

        for idx, file_path in enumerate(files, start=1):
            file_name = os.path.basename(file_path)

            if self.logger:
                self.logger.do_log(
                    f"[RANK] ▶ ({idx}/{len(files)}) file={file_name}",
                    MessageType.INFO,job_id
                )

            # Resolve security by symbol in filename
            security_symbol = None
            for sec in securities:
                if sec.symbol in file_name:
                    security_symbol = sec.symbol
                    break

            if security_symbol is None:
                if self.logger:
                    self.logger.do_log(
                        f"[RANK] ⚠ No security resolved for file={file_name}",
                        MessageType.INFO,job_id
                    )
                continue

            try:
                text = RawFileReader.get_raw_text(file_path)
            except Exception as e:
                if self.logger:
                    self.logger.do_log(
                        f"[RANK] ❌ Failed reading {file_name}: {e}",
                        MessageType.INFO,job_id
                    )
                continue

            chunks = self._get_chunks(text, file_name)
            if not chunks:
                if self.logger:
                    self.logger.do_log(
                        f"[RANK] ❌ No chunks generated for {file_name}",
                        MessageType.INFO,job_id
                    )
                continue

            if self.logger:
                self.logger.do_log(
                    f"[RANK] file={file_name} | security={security_symbol} | chunks={len(chunks)}",
                    MessageType.INFO,job_id
                )



            for chunk_idx, chunk in enumerate(chunks, start=1):
                chunk_emb = self._encode(chunk).squeeze(0)
                max_sim=0.0
                for tag, emb_list in tag_embeddings.items():

                    for we in emb_list:
                        sim = float(torch.dot(chunk_emb, we.squeeze(0)))
                        if sim > max_sim:
                            max_sim = sim

                security_tag_scores[security_symbol].append(max_sim)

                if self.logger:
                    self.logger.do_log(
                        f"[RANK] file={file_name} | chunk={chunk_idx}/{len(chunks)} processed",
                        MessageType.INFO,job_id
                    )

        # Build final results: ONE row per security
        results = []
        for security, tag_arr in security_tag_scores.items():
            ranked_scores=sorted(tag_arr,reverse=True)

            if self.logger:
                self.logger.do_log(
                    f"[RANK] security={security} | ranked_tags={ranked_scores[:top_k]}",
                    MessageType.INFO,job_id
                )

            results.append({
                "security": security,
                "rank_1": ranked_scores[0] if len(ranked_scores) > 0 else None,
                "rank_2": ranked_scores[1] if len(ranked_scores) > 1 else None,
                "rank_3": ranked_scores[2] if len(ranked_scores) > 2 else None,

            })

        # Order by rank_1 score DESC
        results.sort(key=lambda x: x["rank_1"], reverse=True)

        if self.logger:
            self.logger.do_log(
                f"[RANK] ✔ DONE | securities={len(results)}",
                MessageType.INFO,job_id
            )

        self._persist_rank_csv(results, rank_folder,job_id)

        return results




