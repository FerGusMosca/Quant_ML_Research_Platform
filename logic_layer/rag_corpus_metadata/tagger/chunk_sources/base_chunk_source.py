# FILE: logic_layer/rag_corpus_metadata/tagger/chunk_sources/base_chunk_source.py
# Where the tagger gets the chunks of a document from.
#
# Two implementations exist: the filing on disk (chunked and encoded on the
# spot) and the pgvector store (chunks and embeddings already persisted).
# Everything downstream - scoring, density, ranking - is identical either way.

from framework.common.logger.message_type import MessageType


class BaseChunkSource:

    MODE = None          # VECTORS / FILES, shown in every log line
    DESCRIPTION = None   # one line explaining where the data came from

    def __init__(self, tagger, logger=None):
        self.tagger = tagger
        self.logger = logger

    def get_chunks(self, sec_w_file, file_name, fiscal_year, quarter, job_id=None):
        """
        Returns (texts, embeddings):
          texts      - list of chunk strings
          embeddings - a (n_chunks, dim) tensor when the source already has them,
                       or None when the tagger still has to encode the texts.
        Returns ([], None) when this document has nothing usable.
        """
        raise NotImplementedError

    def close(self):
        """Releases whatever the source holds open. No-op by default."""
        pass

    def _log(self, message, job_id=None, level=None):
        if self.logger:
            self.logger.do_log(message, level or MessageType.INFO, job_id)
