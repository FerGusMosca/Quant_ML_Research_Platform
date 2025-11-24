"""
MetadataBuilder
---------------
Attaches metadata to each chunk.
Metadata may include:
 - filename
 - path
 - date
 - type of document
 - ticker (in future)
"""

import os

class MetadataBuilder:

    @staticmethod
    def build(pdf_path, chunk_index):
        return {
            "source_pdf": os.path.basename(pdf_path),
            "chunk_id": chunk_index
        }
