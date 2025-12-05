# multi_stage_rag/pdf_cleaner.py
# All comments MUST be in English.

import re
from rapidfuzz import fuzz
class PDFCleaner:

    @staticmethod
    def _remove_similar_lines(clean_text, logger, threshold=90):
        lines = clean_text.split("\n")
        kept = []

        for idx, line in enumerate(lines):
            line_strip = line.strip()
            if not line_strip:
                continue

            is_dup = False
            for prev in kept:
                sim = fuzz.ratio(line_strip, prev.strip())
                if sim >= threshold:
                    if logger:
                        logger.do_log(f"[MSC] 🔁 Removed similar line #{idx} (sim={sim}): '{line_strip[:80]}'", 2)
                    is_dup = True
                    break

            if not is_dup:
                kept.append(line)

        return "\n".join(kept)

    @staticmethod
    def clean(text: str, logger=None) -> str:
        try:
            if logger:
                logger.do_log("[MSC] 🧹 Starting PDF text cleaning...", 1)

            if text is None or len(text.strip()) == 0:
                if logger: logger.do_log("[MSC] ❌ Empty text received for cleaning.", 0)
                return ""

            # Remove null bytes
            text = text.replace("\x00", "")

            # Remove excessive whitespace
            text = re.sub(r"[ \t]+", " ", text)

            # Normalize line breaks
            text = text.replace("\r", "")
            text = re.sub(r"\n{2,}", "\n", text)

            # Remove common header/footer garbage
            patterns = [
                r"Page \d+ of \d+",
                r"©.*?\d{4}",
                r"Disclaimer.*",
                r"Confidential.*",
            ]
            for p in patterns:
                text = re.sub(p, "", text, flags=re.IGNORECASE)

            clean_text = text.strip()

            # Remove repeated lines (exact duplicates)
            clean_text = PDFCleaner._remove_similar_lines(clean_text, logger)

            if logger:
                logger.do_log("[MSC] ✅ Cleaning completed.", 1)

            return clean_text

        except Exception as e:
            if logger:
                logger.do_log(f"[MSC] ❌ ERROR in PDFCleaner.clean(): {e}", 0)
            return ""
