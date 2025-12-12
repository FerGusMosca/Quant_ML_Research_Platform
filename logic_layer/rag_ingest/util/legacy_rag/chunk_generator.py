"""
VainillaChunkGenerator
--------------
Splits cleaned text into chunks (placeholder version).
Later will include:
 - overlap dynamic
 - semantic segmentation
 - heading-based segmentation
"""

class ChunkGenerator:

    @staticmethod
    def chunk(text: str, chunk_size=1000, overlap=200):
        chunks = []
        words = text.split()

        i = 0
        while i < len(words):
            chunk = words[i:i+chunk_size]
            chunks.append(" ".join(chunk))
            i += chunk_size - overlap

        return chunks
