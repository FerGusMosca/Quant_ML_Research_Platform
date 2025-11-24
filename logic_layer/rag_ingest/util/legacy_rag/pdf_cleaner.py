"""
PDFCleaner
----------
Responsible for normalizing and cleaning text
before chunking (remove artifacts, whitespace, broken lines).
"""

class PDFCleaner:

    @staticmethod
    def clean(text: str) -> str:
        text = text.replace("\x00", "")
        text = text.replace("  ", " ")
        return text.strip()
