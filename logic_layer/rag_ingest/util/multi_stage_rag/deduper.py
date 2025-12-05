# FILE: deduper.py
# All comments MUST be in English.

import hashlib

class ChunkDeduper:
    @staticmethod
    def dedup_chunks(chunks, logger=None):
        """
        Remove exact and near-exact duplicates using hashing.
        Logs every removed duplicate.
        """
        seen = set()
        out = []

        for idx, c in enumerate(chunks):
            # Normalize: trim + collapse whitespace
            norm = " ".join(c.split()).strip()

            # Stable hash
            h = hashlib.md5(norm.encode("utf-8")).hexdigest()

            if h in seen:
                if logger:
                    logger.do_log(f"[DEDUP] 🔁 Removed duplicate chunk #{idx}", 2)
                continue

            seen.add(h)
            out.append(c)

        if logger:
            logger.do_log(f"[DEDUP] ✅ Unique chunks kept: {len(out)} / {len(chunks)}", 1)

        return out
