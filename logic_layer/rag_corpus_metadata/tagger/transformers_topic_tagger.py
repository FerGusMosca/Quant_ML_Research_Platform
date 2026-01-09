# FILE: bert_topic_tagger.py
import json
import os.path

from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

from common.util.extractors.K_Q_10.k_q_10_html_structured_block_extractor import KQ10HtmlStructuredBlockExtractor
from common.util.std_in_out.root_locator import RootLocator
from common.util.tagging.tagging_config_dto import TaggingConfigDTO
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

