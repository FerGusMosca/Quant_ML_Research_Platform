# multi_stage_rag/pdf_text_extractor.py
# All comments MUST be in English.

import fitz

class PDFTextExtractor:

    @staticmethod
    def extract_text(pdf_path: str, logger=None) -> str:
        if logger:
            logger.do_log(f"[MSC] 📄 Extracting PDF: {pdf_path}", 1)

        try:
            with fitz.open(pdf_path) as doc:
                pages = [page.get_text("text") for page in doc]

            text = "\n".join(pages)

            if logger:
                logger.do_log("[MSC] ✅ fitz extraction OK.", 1)

            return text

        except Exception as e:
            if logger:
                logger.do_log(f"[MSC] ❌ fitz extraction failed: {e}", 0)
            return ""
