# FILE: common/util/extractors/section_extractors/k_q_10_section_extractor.py
# Section extractor for SEC 10-K and 10-Q filings.
#
# Thin adapter on top of KQ10HtmlStructuredBlockExtractor: all it does is expose
# that extractor through the interface the vectorization pipeline expects.

from common.util.extractors.K_Q_10.k_q_10_html_structured_block_extractor import (
    KQ10HtmlStructuredBlockExtractor,
)
from common.util.extractors.section_extractors.base_section_extractor import BaseSectionExtractor


class KQ10SectionExtractor(BaseSectionExtractor):

    DOC_TYPE = "K_Q_10"

    def __init__(self):
        self.extractor = KQ10HtmlStructuredBlockExtractor()

    def resolve_sub_type(self, file_name: str) -> str:
        """K10 or Q10, read from the file name (e.g. GPI_2025_Q1_10-Q.html)."""
        return self.extractor.resolve_report_type(file_name)

    def extract_sections(self, raw_text: str, file_name: str) -> dict:
        """
        The narrative items only. Which items those are depends on the report type,
        because a 10-Q numbers MD&A as Part I Item 2 and a 10-K as Item 7.
        """
        report_type = self.resolve_sub_type(file_name)
        blocks = self.extractor.extract_blocks(raw_text, report_type)
        return self.filter_sections(blocks)
