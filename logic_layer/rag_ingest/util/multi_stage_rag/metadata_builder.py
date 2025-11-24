# multi_stage_rag/metadata_builder.py
# Same interface: build(pdf_path, chunk_index)

import os
import datetime

class MetadataBuilder:

    @staticmethod
    def build(pdf_path, chunk_index, extra=None):
        try:
            base = {
                "source_pdf": os.path.basename(pdf_path),
                "chunk_id": chunk_index,
                "ingest_timestamp": datetime.datetime.utcnow().isoformat(),
            }

            if extra:
                base.update(extra)

            return base

        except Exception:
            return {
                "source_pdf": "UNKNOWN",
                "chunk_id": chunk_index,
                "error": "metadata_build_failed"
            }
