# FILE: logic_layer/rag_corpus_metadata/tagger/chunk_sources/chunk_source_factory.py
# Resolves which chunk source a tagging run uses.
#
# VECTORS is the default on purpose: once a filing is vectorized, re-encoding it
# for every tag run is wasted work. FILES stays available for filings that were
# never vectorized, or to compare both paths on the same document.

from framework.common.logger.message_type import MessageType
from logic_layer.rag_corpus_metadata.tagger.chunk_sources.file_chunk_source import FileChunkSource
from logic_layer.rag_corpus_metadata.tagger.chunk_sources.vector_chunk_source import VectorChunkSource


class ChunkSourceFactory:

    VECTORS = "VECTORS"
    FILES = "FILES"
    DEFAULT = VECTORS

    _SOURCES = {
        VECTORS: VectorChunkSource,
        FILES: FileChunkSource,
    }

    @classmethod
    def resolve_mode(cls, chunk_source) -> str:
        """Normalizes the requested mode, falling back to the default when empty."""
        mode = (chunk_source or cls.DEFAULT).upper().strip()

        if mode not in cls._SOURCES:
            raise Exception(
                f"Unknown chunk_source '{chunk_source}'. "
                f"Supported: {', '.join(sorted(cls._SOURCES.keys()))}"
            )
        return mode

    @classmethod
    def build(cls, chunk_source, tagger, logger=None, vectors_db_config=None):
        mode = cls.resolve_mode(chunk_source)

        if mode == cls.VECTORS:
            source = VectorChunkSource(tagger, logger, vectors_db_config)
        else:
            source = FileChunkSource(tagger, logger)

        if logger:
            logger.do_log(
                f"[RANK] ▶ CHUNK SOURCE = {source.MODE} ({source.DESCRIPTION})",
                MessageType.INFO
            )

        return source
