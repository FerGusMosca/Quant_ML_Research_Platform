# FILE: bert_topic_tagger.py
import json
import os.path

from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

from common.util.std_in_out.root_locator import RootLocator
from logic_layer.rag_corpus_metadata.financial_tags import FINANCIAL_TAGS

class TransformersTopicTagger:
    def __init__(self, logger, model_name=None,tag_file=None):
        self.logger = logger
        self.tokenizer = AutoTokenizer.from_pretrained(model_name if model_name is not None else "sentence-transformers/all-MiniLM-L6-v2")
        self.model = AutoModel.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.tag_file=tag_file

        if self.tag_file is None:
            self.keywords = FINANCIAL_TAGS
        else:
            root_path=RootLocator.get_root()
            tag_file=os.path.join(root_path,"static","tags",tag_file)
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

    # ------------------------------------------------------
    def classify(self, text: str):
        """Semantic topic tagging using BERT similarity."""
        text_emb = self._encode(text)
        tags = []

        for topic, words in self.keywords.items():
            sims = []
            for w in words:
                w_emb = self._encode(w)
                sim = float(torch.matmul(text_emb, w_emb.T))
                sims.append(sim)

            if max(sims) > 0.8:   # semantic threshold
                tags.append(topic)

        if not tags:
            tags = ["uncertain"]

        self.logger.do_log(f"[TRANSFORMERS] Tags '{text[:40]}...': {tags}", 2)
        return tags
