"""
PDFTextExtractor
----------------
Responsible for extracting raw text from PDF files.
Uses PyMuPDF (fitz) for high-quality text extraction.
"""

import fitz

class PDFTextExtractor:

    @staticmethod
    def extract_text(pdf_path: str) -> str:
        with fitz.open(pdf_path) as doc:
            pages = [page.get_text("text") for page in doc]
        return "\n".join(pages)
