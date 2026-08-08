from bs4 import BeautifulSoup
import re
from typing import Dict


class KQ10HtmlStructuredBlockExtractor:

    # Any item header at all. Needed to CLOSE the current block: without it,
    # "ITEM 1A" keeps swallowing 1B, 2, 3, 4, 5 and 6 until the next target appears.
    ANY_ITEM_REGEX = re.compile(r"^item\s+(\d{1,2})\s*([a-z])?\b", re.I)

    # Part markers. A 10-Q repeats item numbers across Part I and Part II, so the part
    # is what tells MD&A (Part I Item 2) from Unregistered Sales (Part II Item 2).
    PART_REGEX = re.compile(r"^part\s+(i{1,3}|iv)\b", re.I)

    # Narrative sections worth vectorizing, per report type.
    # Key = (part, item_number, item_letter). part=None means "ignore the part".
    TARGET_ITEMS = {
        "K10": {
            (None, "1", None): "ITEM 1 - BUSINESS",
            (None, "1", "A"): "ITEM 1A - RISK FACTORS",
            (None, "7", None): "ITEM 7 - MD&A",
            (None, "7", "A"): "ITEM 7A - MARKET RISK",
        },
        "Q10": {
            ("I", "2", None): "ITEM 2 - MD&A",
            ("I", "3", None): "ITEM 3 - MARKET RISK",
            ("II", "1", "A"): "ITEM 1A - RISK FACTORS",
        },
    }

    DEFAULT_REPORT_TYPE = "K10"

    @staticmethod
    def resolve_report_type(file_name: str) -> str:
        """Reads the report type out of the filing file name (e.g. GPI_2025_Q1_10-Q.html)."""
        name = (file_name or "").upper()
        if "10-Q" in name or "10Q" in name:
            return "Q10"
        return "K10"

    def _parse_item_header(self, text: str):
        """Returns (number, letter) when the line is an item header, otherwise None."""
        m = self.ANY_ITEM_REGEX.match(text)
        if not m:
            return None
        letter = m.group(2)
        return m.group(1), (letter.upper() if letter else None)

    def extract_blocks(self, html_text: str, report_type: str = None) -> Dict[str, str]:
        """
        Splits a 10-K / 10-Q into its narrative sections.
        Only the items listed in TARGET_ITEMS for the given report type are kept, and
        any other item header closes the block currently being captured.
        """
        report_type = (report_type or self.DEFAULT_REPORT_TYPE).upper()
        targets = self.TARGET_ITEMS.get(report_type, self.TARGET_ITEMS[self.DEFAULT_REPORT_TYPE])

        soup = BeautifulSoup(html_text, "lxml")

        for tag in soup.find_all(["ix:header", "ix:nonNumeric", "ix:nonFraction"]):
            tag.decompose()

        body = soup.body
        if not body:
            return {}

        blocks = {}
        state = {"label": None, "buffer": []}
        current_part = None

        def flush():
            # The table of contents produces a tiny block carrying the same label as the
            # real section, so the longest capture per label is the one that survives.
            if state["label"] and state["buffer"]:
                text = " ".join(state["buffer"]).strip()
                if len(text) > len(blocks.get(state["label"], "")):
                    blocks[state["label"]] = text

        for el in body.stripped_strings:
            # Normalize unicode spaces before matching
            el_normalized = re.sub(r'[\xa0\u200b\s]+', ' ', el).strip()
            if not el_normalized:
                continue

            part_match = self.PART_REGEX.match(el_normalized)
            if part_match:
                current_part = part_match.group(1).upper()
                continue

            parsed = self._parse_item_header(el_normalized)
            if parsed:
                number, letter = parsed
                label = targets.get((current_part, number, letter)) or targets.get((None, number, letter))

                flush()
                state["buffer"] = []
                state["label"] = label  # None when the item is not a target -> capture stops
                continue

            if state["label"]:
                state["buffer"].append(el_normalized)

        flush()

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