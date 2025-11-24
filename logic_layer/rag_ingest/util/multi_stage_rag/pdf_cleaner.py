# multi_stage_rag/pdf_cleaner.py
# All comments MUST be in English.

import re

class PDFCleaner:

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

            if logger:
                logger.do_log("[MSC] ✅ Cleaning completed.", 1)

            return clean_text

        except Exception as e:
            if logger:
                logger.do_log(f"[MSC] ❌ ERROR in PDFCleaner.clean(): {e}", 0)
            return ""
