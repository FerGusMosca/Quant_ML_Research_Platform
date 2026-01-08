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
        try:
            try:
                doc = fitz.open(pdf_path)
            except Exception as e:
                self.logger.do_log(f"[HASH] ❌ Corrupted PDF skipped: {pdf_path} -- {e}", 1)
                return None, None, True

            text = ""
            try:
                text = doc[0].get_text("text")
            except:
                pass

            try:
                text_hash = hashlib.md5(text.encode("utf-8", errors="ignore")).hexdigest()
            except:
                text_hash = None

            try:
                with open(pdf_path, "rb") as f:
                    file_hash = hashlib.md5(f.read()).hexdigest()
            except:
                file_hash = None

            return text_hash, file_hash, False

        except Exception as e:
            self.logger.do_log(f"[HASH] ❌ Unexpected hashing error: {pdf_path} -- {e}", 1)
            return None, None, True

