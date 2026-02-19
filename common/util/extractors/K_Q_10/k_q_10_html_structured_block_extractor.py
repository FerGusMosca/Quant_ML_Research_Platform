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
            # Normalize unicode spaces before matching
            el_normalized = re.sub(r'[\xa0\u200b\s]+', ' ', el).strip()

            if self.ITEM_REGEX.match(el_normalized):
                if current_item and buffer:
                    blocks[current_item] = " ".join(buffer).strip()
                current_item = el_normalized.upper()
                buffer = []
            else:
                if current_item:
                    buffer.append(el_normalized)

        if current_item and buffer:
            blocks[current_item] = " ".join(buffer).strip()

        return blocks

    def extract_blocks_adv(self, html_text: str, sections: list[str]) -> dict[str, str]:
        from sec_parser import Edgar10QParser
        from bs4 import BeautifulSoup
        import re

        # Pre-process iXBRL: strip inline XBRL tags before parsing
        # This ensures sec_parser gets clean HTML even for 10-K iXBRL filings
        soup = BeautifulSoup(html_text, "lxml")
        for tag in soup.find_all(re.compile(r'^ix:')):
            tag.unwrap()
        clean_html = str(soup)

        parser = Edgar10QParser()
        semantic_tree = parser.parse(clean_html)

        sections_upper = {s.upper(): s for s in sections}
        blocks: dict[str, list[str]] = {}
        current_key: str | None = None
        section_counter = 0

        for node in semantic_tree:
            text = node.text.strip()
            if not text: continue
            upper = text.upper()

            is_new_main_header = re.match(r'^\s*(ITEM\s+\d+|PART\s+[IV]+)', upper)
            matched_section = next((s for s in sections_upper if s in upper), None)

            if matched_section and is_new_main_header:
                if len(text) > 200 or "...." in text:
                    continue
                section_counter += 1
                current_key = f"{matched_section}_{section_counter}"
                blocks[current_key] = []
                continue

            if current_key:
                if is_new_main_header and not any(s in upper for s in sections_upper):
                    current_key = None
                    continue
                blocks[current_key].append(text)

        blocks_text = {f"{k}_AGGR": "\n".join(v).strip() for k, v in blocks.items()}
        return blocks, blocks_text

    def extract_blocks_by_item(self, html_text: str, item_titles: list[str], report_type: str = "Q10",
                               skip_tables: bool = True) -> dict[str, str]:
        try:
            if report_type == "K10":
                from sec_parser import Edgar10KParser
                parser = Edgar10KParser()
            else:
                from sec_parser import Edgar10QParser
                parser = Edgar10QParser()
        except ImportError:
            # If Edgar10KParser is not available, fallback to Edgar10QParser
            from sec_parser import Edgar10QParser
            parser = Edgar10QParser()

        from sec_parser.semantic_elements import TopSectionTitle, TableElement

        semantic_tree = parser.parse(html_text)

        # Normalize target titles for matching
        titles_upper = [t.upper() for t in item_titles]

        blocks: dict[str, list[str]] = {}
        current_key: str | None = None
        section_counter = 0

        for node in semantic_tree:
            text = node.text.strip()
            if not text:
                continue
            upper = text.upper()

            # Skip table nodes entirely if skip_tables is enabled
            # This prevents numeric table rows from polluting the extracted text
            if skip_tables and isinstance(node, TableElement):
                continue

            is_real_header = isinstance(node, TopSectionTitle)

            if is_real_header:
                if any(title in upper for title in titles_upper):
                    # Start capturing — matched our target item
                    section_counter += 1
                    current_key = f"{upper.strip()}_{section_counter}"
                    blocks[current_key] = []

                elif current_key:
                    # Any other real header → stop capturing
                    current_key = None

                # Headers are never added to block content
                continue

            # Regular content node → append only if we are inside a target section
            if current_key:
                blocks[current_key].append(text)

        # Aggregate each block's lines into a single string
        blocks_text = {f"{k}_AGGR": "\n".join(v).strip() for k, v in blocks.items()}
        return blocks, blocks_text