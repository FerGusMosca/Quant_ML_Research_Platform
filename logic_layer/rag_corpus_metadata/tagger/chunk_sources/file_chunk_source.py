# FILE: logic_layer/rag_corpus_metadata/tagger/chunk_sources/file_chunk_source.py
# Chunks read straight from the filing on disk.
#
# This is the original behaviour: extract the narrative items, split them into
# ~180 word chunks and let the tagger encode them. Slower, but it needs nothing
# in the database and works on files that were never vectorized.

from logic_layer.rag_corpus_metadata.tagger.chunk_sources.base_chunk_source import BaseChunkSource


class FileChunkSource(BaseChunkSource):

    MODE = "FILES"
    DESCRIPTION = "chunks extracted from the filing on disk and encoded on the fly"

    def get_chunks(self, sec_w_file, file_name, fiscal_year, quarter, job_id=None):
        text = self.tagger._extract_text(sec_w_file.file, job_id)
        if not text:
            return [], None

        chunks = self.tagger._extract_chunks(text, file_name, job_id)
        if not chunks:
            return [], None

        # No embeddings: the tagger encodes these itself
        return chunks, None
