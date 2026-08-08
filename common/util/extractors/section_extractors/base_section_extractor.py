# FILE: common/util/extractors/section_extractors/base_section_extractor.py
# Contract every document family must satisfy to be vectorizable.
#
# Adding a new document family (earnings call transcripts, PDFs, 8-K, press
# releases) means writing one subclass and registering it. No if/else in the
# orchestration layer has to change.


class BaseSectionExtractor:

    DOC_TYPE = None          # registry key, e.g. K_Q_10
    MIN_SECTION_CHARS = 200  # sections shorter than this are noise ('None.', TOC leftovers)

    def resolve_sub_type(self, file_name: str) -> str:
        """
        Narrows the family down to the concrete document, e.g. K_Q_10 -> K10 or Q10.
        The value is stored in filing_documents.report_type.
        """
        raise NotImplementedError

    def extract_sections(self, raw_text: str, file_name: str) -> dict:
        """
        Returns {section_label: section_text} with the parts worth vectorizing.
        Labels are free text but should stay stable across runs: they are what
        the semantic search filters on later.
        """
        raise NotImplementedError

    def filter_sections(self, sections: dict) -> dict:
        """Drops sections too short to carry signal. Shared by every family."""
        return {
            label: text
            for label, text in (sections or {}).items()
            if text and len(text) >= self.MIN_SECTION_CHARS
        }
