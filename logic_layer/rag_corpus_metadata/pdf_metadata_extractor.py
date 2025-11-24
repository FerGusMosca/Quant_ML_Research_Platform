# All comments MUST be in English.

import fitz  # PyMuPDF
import os

# Disable noisy MuPDF console errors
fitz.TOOLS.mupdf_display_errors(False)

class PDFMetadataExtractor:

    def __init__(self, logger):
        self.logger = logger

    def extract(self, pdf_path):
        doc = fitz.open(pdf_path)

        # --- pages ---
        pages = len(doc)

        # --- first page text only (TURBO MODE) ---
        try:
            first_page = doc[0].get_text("text")
        except:
            first_page = ""

        # --- text length without reading the whole PDF ---
        # We approximate text length using only first page.
        # True full-text is NOT needed for metadata pipeline.
        text_length = len(first_page)

        # --- title guess ---
        title_guess = first_page.split("\n")[0][:180].strip()

        return {
            "path": pdf_path,
            "filename": os.path.basename(pdf_path),
            "folder": os.path.dirname(pdf_path),
            "pages": pages,
            "text_length": text_length,
            "title_guess": title_guess,
        }
