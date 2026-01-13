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
            self._log_exception("[RANK] ❌ _persist_rank_csv failed", e, job_id)

    def _log_exception(self, prefix: str, exc: Exception, job_id: int):
        tb = traceback.extract_tb(exc.__traceback__)
        last = tb[-1]
        msg = f"{prefix} | {exc.__class__.__name__}: {exc} | line={last.lineno} | file={last.filename}"
        if self.logger:
            self.logger.do_log(msg, MessageType.ERROR, job_id)

    def _extract_text(self, file_path: str, job_id: int):
        try:
            return RawFileReader.get_raw_text(file_path)
        except Exception as e:
            self._log_exception("[RANK] ❌ extract_text failed", e, job_id)
            return None

    def _extract_chunks(self, text: str, file_name: str, job_id: int):
        try:
            chunks = self._get_chunks(text, file_name)
            if not chunks:
                if self.logger:
                    self.logger.do_log(
                        f"[RANK] ❌ No chunks generated | file={file_name}",
                        MessageType.INFO, job_id
                    )
            return chunks
        except Exception as e:
            self._log_exception("[RANK] ❌ extract_chunks failed", e, job_id)
            return None

    def _score_chunks(
            self,
            chunks: list,
            tag_embeddings: dict,
            file_name: str,
            security_symbol: str,
            job_id: int
    ):
        scores = []

        for chunk_idx, chunk in enumerate(chunks, start=1):
            try:
                chunk_emb = self._encode(chunk).squeeze(0)
                max_sim = 0.0

                for emb_list in tag_embeddings.values():
                    for we in emb_list:
                        sim = float(torch.dot(chunk_emb, we.squeeze(0)))
                        if sim > max_sim:
                            max_sim = sim

                scores.append(max_sim)

                if self.logger:
                    self.logger.do_log(
                        f"[RANK] file={file_name} | security={security_symbol} | chunk={chunk_idx}/{len(chunks)} processed",
                        MessageType.INFO, job_id
                    )

            except Exception as e:
                self._log_exception("[RANK] ❌ scoring failed", e, job_id)

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

    def initialize_tag_dict(self,job_id):
        try:

            tag_dict = {}
            if self.tag_cfg.tag_file is not None:

                tag_dict = JsonFileReader.load_json_file(
                    os.path.join(RootLocator.get_root(), "static", "tags"),self.tag_cfg.tag_file)
                self.logger.do_log(f"[TAGGING] Loading tag file in tag folder static/tags  file={self.tag_cfg.tag_file}",
                                   MessageType.INFO, job_id)
            elif self.tag_cfg.tag_json is not None:
                tag_dict = json.loads(self.tag_cfg.tag_json)
                self.logger.do_log(f"[TAGGING] Loading tag file from tag_json field!  keys={tag_dict.keys()}",
                                   MessageType.INFO, job_id)
            else:
                raise Exception("Missing tag_file or tag_json in the input file")

            return tag_dict
        except Exception as e:
            #self._log_exc(f"[TAGGING] ❌ tag load failed | year={y}", e, job_id)
            raise e


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





