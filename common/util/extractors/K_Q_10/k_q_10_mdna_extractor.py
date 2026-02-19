import re
from typing import Optional
from bs4 import BeautifulSoup
from common.util.extractors.K_Q_10.k_q_10_html_structured_block_extractor import KQ10HtmlStructuredBlockExtractor
from framework.common.logger.message_type import MessageType


class KQ10MDNAExtractor:
    """
    Extracts Management's Discussion and Analysis (MD&A) section
    from SEC 10-K (annual) and 10-Q (quarterly) filings.

    10-K: MD&A is located in Part II, Item 7
    10-Q: MD&A is located in Part I, Item 2

    Usage:
        extractor = KQ10MDNAExtractor(logger)
        mdna_text = extractor.extract(text, symbol="AAPL", form_type="10-K")
    """

    # 10-K patterns: Part II, Item 7 -> ends at Item 7A or Item 8
    _10K_START_PATTERNS = [
        r"Part\s+II[,.\s]+Item\s*7[^A]",
        r"Part\s+II[,.\s]+Item\s*7\b",
        r"Item\s*7[.:\s]+Management[''']?s?\s+Discussion",
        r"Item\s*7[.:\s]+MD&A",
        r"MANAGEMENT[''']?S?\s+DISCUSSION\s+AND\s+ANALYSIS\s+OF\s+FINANCIAL\s+CONDITION",
        r"Management[''']?s?\s+Discussion\s+and\s+Analysis\s+of\s+Financial\s+Condition",
        r"Item\s*7\s*[-–—]\s*Management",
    ]

    _10K_STOP_PATTERNS = [
        r"Item\s*7A[.:\s]+Quantitative",
        r"Part\s+II[,.\s]+Item\s*7A",
        r"Item\s*8[.:\s]+Financial\s+Statements",
        r"Part\s+II[,.\s]+Item\s*8",
        r"QUANTITATIVE\s+AND\s+QUALITATIVE\s+DISCLOSURES",
        r"Report\s+of\s+Independent\s+Registered",
        r"CONSOLIDATED\s+FINANCIAL\s+STATEMENTS",
    ]

    _10K_CONFIG = {
        "min_length": 5000,
        "fallback_length": 30000,
        "search_window": 50000,
        "form_type": "10-K"
    }

    # 10-Q patterns: Part I, Item 2 -> ends at Item 3 or Item 4
    _10Q_START_PATTERNS = [
        # Matches Item 2 only at the start of a line to avoid TOC tables
        r"(?m)^\s*Item\s*2[.:\s]+Management",
        r"(?i)Item\s*2[.:\s]+Management[''']?s?\s+Discussion\s+and\s+Analysis",
        r"(?i)Forward-Looking\s+Statements\s+and\s+Factors\s+That\s+May\s+Affect\s+Future\s+Results",
    ]

    _10Q_STOP_PATTERNS = [
        r"Item\s*3[.:\s]*Quantitative",
        r"ITEM\s*3[.:\s]*QUANTITATIVE",
        r"Part\s+I[,.\s]+Item\s*3",
        r"PART\s+I[,.\s]+ITEM\s*3",
        r"Item\s*4[.:\s]*Controls",
        r"ITEM\s*4[.:\s]*CONTROLS",
        r"QUANTITATIVE\s+AND\s+QUALITATIVE\s+DISCLOSURES",
        r"CONTROLS\s+AND\s+PROCEDURES",
        r"Part\s+II[.\s]",
        r"PART\s+II[.\s]",
    ]

    _10Q_CONFIG = {
        "min_length": 2000,
        "fallback_length": 15000,
        "search_window": 30000,
        "form_type": "10-Q"
    }

    # Fallback keyword pattern for last-resort extraction
    _FALLBACK_KEYWORDS = re.compile(
        r"\b(results\s+of\s+operations|revenue\s+(increased|decreased)|"
        r"operating\s+income|net\s+sales|financial\s+condition)\b",
        re.I
    )

    def __init__(self, logger):
        """
        Initialize the MD&A extractor.

        Args:
            logger: Logger instance with do_log(message, MessageType) method
        """
        self.logger = logger

    def extract(self, text: str, symbol: str, form_type: str = "10-K") -> str:
        """
        Extract MD&A section from filing text.

        Args:
            text: Raw text content of the SEC filing
            symbol: Stock ticker symbol (for logging)
            form_type: "10-K" or "10-Q"

        Returns:
            Extracted MD&A text, or empty string if extraction fails
        """
        from framework.common.logger.message_type import MessageType

        form_normalized = form_type.upper().replace("-", "").replace(" ", "")

        if "10K" in form_normalized:
            return self._extract_10k(text, symbol)
        elif "10Q" in form_normalized:
            return self._extract_10q(text, symbol)
        else:
            self.logger.do_log(
                f"[MDNA][{symbol}] Unknown form type '{form_type}', defaulting to 10-K",
                MessageType.WARNING
            )
            return self._extract_10k(text, symbol)

    def _skip_toc(self, text: str) -> str:
        """Skip Table of Contents section to avoid false matches."""
        toc_end_patterns = [
            r"TABLE\s+OF\s+CONTENTS.*?(?=Item\s*1[.\s])",
            r"INDEX.*?(?=Part\s+I\s+Item\s*1)",
        ]

        for pattern in toc_end_patterns:
            match = re.search(pattern, text[:50000], re.I | re.DOTALL)
            if match:
                return text[match.end():]

        return text

    def _is_table(self, text_block: str, threshold: float = 0.20) -> bool:
        """
        Detects if a text block is likely a financial table rather than narrative.
        Checks the density of digits in the first 1000 characters.
        """
        if not text_block or len(text_block) < 200:
            return False

        sample = text_block[:1000]
        digit_count = sum(c.isdigit() for c in sample)
        return (digit_count / len(sample)) > threshold

    def _extract_10k(self, text: str, symbol: str) -> str:
        from framework.common.logger.message_type import MessageType

        try:
            html_extractor = KQ10HtmlStructuredBlockExtractor()
            # En 10-K el MD&A es el Item 7
            _, blocks_text = html_extractor.extract_blocks_adv(
                text,
                ["MANAGEMENT’S DISCUSSION AND ANALYSIS",
                 "MANAGEMENT'S DISCUSSION AND ANALYSIS",
                 "MANAGEMENTS DISCUSSION AND ANALYSIS"]
            )

            mdna = "".join(blocks_text.values())

            if len(mdna) > self._10K_CONFIG["min_length"]:
                self.logger.do_log(
                    f"[MDNA][{symbol}][10-K] extract_blocks_adv OK: {len(mdna):,} chars",
                    MessageType.DEBUG
                )
                return mdna

        except Exception as e:
            self.logger.do_log(f"[MDNA][{symbol}][10-K] Cartógrafo falló: {e}", MessageType.WARNING)

        # Fallback
        return self._do_extract(
            text=text,
            symbol=symbol,
            start_patterns=self._10K_START_PATTERNS,
            stop_patterns=self._10K_STOP_PATTERNS,
            config=self._10K_CONFIG
        )

    def _extract_10q(self, text: str, symbol: str) -> str:
        from framework.common.logger.message_type import MessageType

        try:
            html_extractor = KQ10HtmlStructuredBlockExtractor()
            _, blocks_text = html_extractor.extract_blocks_by_item(
                text,
                item_titles=["ITEM 2"],
                skip_tables=True
            )

            mdna=""
            for text in blocks_text.values():
                mdna+=text

            if len(mdna) > self._10Q_CONFIG["min_length"]:
                self.logger.do_log(
                    f"[MDNA][{symbol}][10-Q] extract_blocks_adv OK: {len(mdna):,} chars",
                    MessageType.DEBUG
                )
                return mdna

            self.logger.do_log(
                f"[MDNA][{symbol}][10-Q] extract_blocks_adv too short, falling back",
                MessageType.WARNING
            )

        except Exception as e:
            self.logger.do_log(
                f"[MDNA][{symbol}][10-Q] extract_blocks_adv exception: {e}, falling back",
                MessageType.WARNING
            )

        plain_text = BeautifulSoup(text, "lxml").get_text(" ", strip=True)
        plain_text = self._skip_toc(plain_text)
        return self._do_extract(
            text=plain_text,
            symbol=symbol,
            start_patterns=self._10Q_START_PATTERNS,
            stop_patterns=self._10Q_STOP_PATTERNS,
            config=self._10Q_CONFIG
        )
    def _do_extract(
            self,
            text: str,
            symbol: str,
            start_patterns: list,
            stop_patterns: list,
            config: dict,
            depth: int = 0
    ) -> str:
        """Core extraction logic with table-skipping recursion."""
        from framework.common.logger.message_type import MessageType

        # Safety break to prevent infinite loops
        if depth > 5 or len(text) < 1000:
            return ""

        form_type = config["form_type"]
        text = re.sub(r'\s+', ' ', text)

        # Find start position
        start_pos = self._find_start(text, symbol, form_type, start_patterns)
        if start_pos is None:
            return ""

        # Find stop position
        stop_pos = self._find_stop(
            text, symbol, form_type, stop_patterns,
            start_pos, config["search_window"]
        )

        # Extract raw content
        mdna_text = text[start_pos:stop_pos].strip()

        # Table detection and skip logic
        if self._is_table(mdna_text):
            self.logger.do_log(
                f"[MDNA][{symbol}] Table/Noise detected at initial match, searching deeper (depth {depth})...",
                MessageType.WARNING
            )
            # Recurse by skipping the current table match
            return self._do_extract(text[start_pos + 100:], symbol, start_patterns, stop_patterns, config, depth + 1)

        # Minimum length validation and fallback
        if len(mdna_text) < config["min_length"]:
            self.logger.do_log(
                f"[MDNA][{symbol}][{form_type}] MD&A too short ({len(mdna_text):,} chars), using fallback",
                MessageType.WARNING
            )
            mdna_text = text[start_pos:start_pos + config["fallback_length"]]
        else:
            self.logger.do_log(
                f"[MDNA][{symbol}][{form_type}] MD&A successfully extracted: {len(mdna_text):,} chars",
                MessageType.DEBUG
            )

        return re.sub(r'\s+', ' ', mdna_text).strip()

    def _find_start(
            self,
            text: str,
            symbol: str,
            form_type: str,
            patterns: list
    ) -> Optional[int]:
        """Find the start position of MD&A section."""
        from framework.common.logger.message_type import MessageType

        start_pos = len(text)
        matched_pattern = None

        for pattern in patterns:
            match = re.search(pattern, text, re.I)
            if match and match.start() < start_pos:
                start_pos = match.start()
                matched_pattern = pattern[:40]

        if start_pos < len(text):
            self.logger.do_log(
                f"[MDNA][{symbol}][{form_type}] START: '{matched_pattern}' @ {start_pos:,}",
                MessageType.DEBUG
            )
            return start_pos

        # Fallback
        self.logger.do_log(
            f"[MDNA][{symbol}][{form_type}] No start pattern matched, trying fallback",
            MessageType.WARNING
        )

        fallback_match = self._FALLBACK_KEYWORDS.search(text[:150000])
        if fallback_match:
            fallback_pos = max(0, fallback_match.start() - 2000)
            self.logger.do_log(
                f"[MDNA][{symbol}][{form_type}] FALLBACK start @ {fallback_pos:,}",
                MessageType.WARNING
            )
            return fallback_pos

        self.logger.do_log(
            f"[MDNA][{symbol}][{form_type}] EXTRACTION FAILED - no patterns found",
            MessageType.ERROR
        )
        return None

    def _find_stop(
            self,
            text: str,
            symbol: str,
            form_type: str,
            patterns: list,
            start_pos: int,
            search_window: int
    ) -> int:
        """Find the stop position of MD&A section."""
        from framework.common.logger.message_type import MessageType

        tail = text[start_pos:start_pos + search_window]
        stop_pos = start_pos + len(tail)
        matched_pattern = "END-OF-WINDOW"

        for pattern in patterns:
            match = re.search(pattern, tail, re.I)
            if match:
                candidate = start_pos + match.start()
                if candidate < stop_pos:
                    stop_pos = candidate
                    matched_pattern = pattern[:30]

        self.logger.do_log(
            f"[MDNA][{symbol}][{form_type}] STOP: '{matched_pattern}' @ {stop_pos:,}",
            MessageType.DEBUG
        )

        return stop_pos