from logic_layer.rag_corpus_metadata.financial_tags import FINANCIAL_TAGS

class TopicTagger:
    def __init__(self):
        self.keywords = FINANCIAL_TAGS

    def classify(self, text):
        text_l = text.lower()
        tags = []
        for topic, words in self.keywords.items():
            if any(w in text_l for w in words):
                tags.append(topic)
        return tags
