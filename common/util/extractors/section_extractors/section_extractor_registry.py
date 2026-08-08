# FILE: common/util/extractors/section_extractors/section_extractor_registry.py
# Maps a doc_type to the extractor that knows how to read that family.
#
# This is the extension point. To support earnings call transcripts tomorrow:
#
#   1) write TranscriptSectionExtractor(BaseSectionExtractor) with DOC_TYPE = "TRANSCRIPT"
#   2) add one line to _EXTRACTORS below
#
# Nothing in the orchestration layer or in the persistence layer changes.

from common.util.extractors.section_extractors.k_q_10_section_extractor import KQ10SectionExtractor


class SectionExtractorRegistry:

    _EXTRACTORS = {
        KQ10SectionExtractor.DOC_TYPE: KQ10SectionExtractor,
        # "TRANSCRIPT": TranscriptSectionExtractor,
        # "PDF_REPORT": PdfReportSectionExtractor,
    }

    @classmethod
    def get(cls, doc_type: str):
        """Returns a ready extractor instance, or raises with the supported list."""
        key = (doc_type or "").upper().strip()
        extractor_class = cls._EXTRACTORS.get(key)

        if extractor_class is None:
            raise Exception(
                f"No section extractor registered for doc_type='{doc_type}'. "
                f"Supported: {', '.join(sorted(cls._EXTRACTORS.keys()))}"
            )

        return extractor_class()

    @classmethod
    def supported_doc_types(cls) -> list:
        return sorted(cls._EXTRACTORS.keys())

    @classmethod
    def register(cls, extractor_class):
        """Optional runtime registration, for extractors living outside this module."""
        cls._EXTRACTORS[extractor_class.DOC_TYPE] = extractor_class
