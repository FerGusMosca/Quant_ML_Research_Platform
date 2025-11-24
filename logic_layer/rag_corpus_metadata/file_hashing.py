# file_hashing.py
import hashlib
import fitz


class FileHashing:

    def __init__(self, logger):
        self.logger = logger

    def _hash_bytes(self, data):
        h = hashlib.sha256()
        h.update(data)
        return h.hexdigest()

    def compute_hashes(self, pdf_path):
        with open(pdf_path, "rb") as f:
            file_bytes = f.read()
        file_hash = self._hash_bytes(file_bytes)

        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text("text")
        text_hash = self._hash_bytes(text.encode("utf-8"))

        return text_hash, file_hash
