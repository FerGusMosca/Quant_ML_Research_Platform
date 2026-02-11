import os
import re

from bs4 import BeautifulSoup


class F4Processor:
    """
    Extracts insider trading details from Form 4 HTML filings.
    """
    def process_file(self, file_path: str) -> dict:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            html_content = f.read()

        soup = BeautifulSoup(html_content, 'html.parser')
        text_content = soup.get_text(" ", strip=True)

        # 1. Metadata: Identify the Insider
        # Form 4s usually have "1. Name and Address of Reporting Person"
        insider_name = "Unknown"
        name_section = soup.find(string=re.compile(r"1\. Name and Address of Reporting Person", re.I))
        if name_section:
            # Logic to grab the next relevant text block
            insider_name = name_section.find_next().get_text(strip=True)

        # 2. Extract Relationships (Director, 10% Owner, Officer)
        is_director = "X" in (soup.find(string=re.compile(r"Director", re.I)) or "")
        is_officer = "X" in (soup.find(string=re.compile(r"Officer", re.I)) or "")

        # 3. Clean Content
        for tag in soup(['style', 'script', 'link', 'meta']):
            tag.decompose()

        clean_text = soup.get_text(separator='\n', strip=True)
        clean_text = re.sub(r'\n+', '\n', clean_text)

        return {
            "metadata": {
                "insider_name": insider_name,
                "is_director": is_director,
                "is_officer": is_officer
            },
            "clean_text": clean_text
        }