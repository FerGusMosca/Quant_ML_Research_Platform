import fitz
import os

fitz.TOOLS.mupdf_display_errors(False)

class PDFMetadataExtractor:
    def __init__(self, logger):
        self.logger = logger

    def extract(self, pdf_path):
        try:
            # --- File exists ---
            if not os.path.exists(pdf_path):
                self.logger.do_log(f"[META] ❌ File not found: {pdf_path}", 1)
                return {"path": pdf_path, "skipped": True}

            # --- File > 0 bytes ---
            if os.path.getsize(pdf_path) < 32:
                self.logger.do_log(f"[META] ❌ Empty or corrupt file: {pdf_path}", 1)
                return {"path": pdf_path, "skipped": True}

            # --- Attempt open ---
            try:
                doc = fitz.open(pdf_path)
            except Exception as e:
                self.logger.do_log(f"[META] ❌ Could not open PDF: {pdf_path} | {e}", 1)
                return {"path": pdf_path, "skipped": True}

            # --- Pages ---
            try:
                pages = len(doc)
            except:
                pages = 0

            # --- First page ---
            try:
                first_page = doc[0].get_text("text") if pages > 0 else ""
            except:
                first_page = ""

            text_length = len(first_page)
            title_guess = first_page.split("\n")[0][:180].strip() if first_page else ""

            return {
                "path": pdf_path,
                "filename": os.path.basename(pdf_path),
                "folder": os.path.dirname(pdf_path),
                "pages": pages,
                "text_length": text_length,
                "title_guess": title_guess,
                "skipped": False
            }

        except Exception as e:
            self.logger.do_log(f"[META] ❌ Fatal metadata error: {pdf_path} | {e}", 1)
            return {"path": pdf_path, "skipped": True}
