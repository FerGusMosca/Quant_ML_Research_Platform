import os
import re
from bs4 import BeautifulSoup


class K8Processor:
    """
    Handles the extraction of clean text and metadata from SEC HTML filings.
    """

    def process_file(self, file_path: str) -> dict:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            html_content = f.read()

        # Using lxml for speed; falls back to html.parser if not installed
        soup = BeautifulSoup(html_content, 'html.parser')

        # 1. Metadata extraction: Items (e.g., Item 2.02, Item 9.01)
        # We search for the pattern in the text
        text_content = soup.get_text(" ", strip=True)
        items = sorted(list(set(re.findall(r"Item\s+(\d+\.\d+)", text_content))))

        # 2. Extract Date from filename or content
        # Filename format: SYMBOL_YYYY-MM-DD_8-K.html
        date_match = re.search(r"(\d{4}-\d{2}-\d{2})", os.path.basename(file_path))
        report_date = date_match.group(1) if date_match else "unknown"

        # 3. Clean Content for the UI/LLM
        # Remove noise tags that don't carry financial information
        for tag in soup(['style', 'script', 'ix:header', 'link', 'meta']):
            tag.decompose()

        # Get full structured text
        clean_text = soup.get_text(separator='\n', strip=True)

        # Remove multiple newlines for a cleaner UI display
        clean_text = re.sub(r'\n+', '\n', clean_text)

        return {
            "metadata": {
                "date": report_date,
                "items": items
            },
            "clean_text": clean_text  # This is the REAL body content
        }