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

    def extract_blocks_adv(self, html_text: str, sections: list[str]) -> dict[str, str]:
        """
        Extract multiple sections (e.g. COMPETITION, RISK FACTORS) from a 10-Q semantic tree.
        Each section is captured independently and stops when a new header is detected.
        """
        from sec_parser import Edgar10QParser

        parser = Edgar10QParser()
        semantic_tree = parser.parse(html_text)

        sections_upper = {s.upper(): s for s in sections}  # normalize
        blocks: dict[str, list[str]] = {}

        current_section: str | None = None

        for node in semantic_tree:
            text = node.text.strip()
            if not text:
                continue

            upper = text.upper()

            matched_section = next((s for s in sections_upper if s in upper), None)
            # Detect section header
            if matched_section:
                #current_section = sections_upper[upper]
                current_section=matched_section
                blocks[current_section] = []
                continue

            # If we are inside a tracked section
            if current_section:
                # Heuristic to detect next header and close current section
                if len(text.split()) <= 3 and not text.endswith(".") and text.isalpha():
                    current_section = None
                    continue

                blocks[current_section].append(text)

        # Join text blocks
        blocks_text={}
        for k, v in blocks.items():
            blocks_text[k+"_AGGR"]="\n".join(v).strip()

        return blocks,blocks_text



