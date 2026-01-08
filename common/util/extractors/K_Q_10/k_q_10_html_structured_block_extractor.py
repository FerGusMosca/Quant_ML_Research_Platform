from bs4 import BeautifulSoup
import re
from typing import Dict


class KQ10HtmlStructuredBlockExtractor:

    ITEM_REGEX = re.compile(
        r"(item\s+1a\.?|item\s+1\.?|item\s+7a\.?|item\s+7\.?)",
        re.I
    )

    def extract_blocks(self, html_text: str) -> Dict[str, str]:
        soup = BeautifulSoup(html_text, "lxml")

        for tag in soup.find_all(["ix:header", "ix:nonNumeric", "ix:nonFraction"]):
            tag.decompose()

        body = soup.body
        if not body:
            return {}

        blocks = {}
        current_item = None
        buffer = []

        for el in body.stripped_strings:
            if self.ITEM_REGEX.match(el):
                if current_item and buffer:
                    blocks[current_item] = " ".join(buffer).strip()
                current_item = el.upper()
                buffer = []
            else:
                if current_item:
                    buffer.append(el)

        if current_item and buffer:
            blocks[current_item] = " ".join(buffer).strip()

        return blocks
