# corpus_metadata_pipeline.py
import os
from tqdm import tqdm

from logic_layer.rag_corpus_metadata.drift_detector import DriftDetector
from logic_layer.rag_corpus_metadata.file_hashing import FileHashing
from logic_layer.rag_corpus_metadata.metadata_inventory_builder import MetadataInventoryBuilder
from logic_layer.rag_corpus_metadata.pdf_metadata_extractor import PDFMetadataExtractor


class CorpusMetadataPipeline:

    def __init__(self, config, logger, dest_root):
        self.config = config
        self.logger = logger
        self.dest_root = dest_root

        folder = os.path.join(config["RAG_OUTPUT_FOLDER"], "corpus_metadata")
        os.makedirs(folder, exist_ok=True)
        self.output_folder = folder

        self.extractor = PDFMetadataExtractor(logger)
        self.hasher = FileHashing(logger)
        self.drift = DriftDetector(logger)
        self.inventory = MetadataInventoryBuilder(folder, logger)

    def run(self, pdf_list):
        metadata_items = []

        for pdf in tqdm(pdf_list):
            text_hash, file_hash = self.hasher.compute_hashes(pdf)
            meta = self.extractor.extract(pdf)

            meta["sha256_file"] = file_hash
            meta["sha256_text"] = text_hash
            meta["status"] = "unknown"

            metadata_items.append(meta)

        final_items = self.drift.apply_status(metadata_items)
        self.inventory.save(final_items)
