# FILE: deduper.py
# All comments MUST be in English.

import hashlib

class VanillaChunkDeduper:

    def __init__(self,logger):
        self.logger=logger

    def dedup_chunks(self,chunks):
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
                if self.logger:
                    self.logger.do_log(f"[DEDUP] 🔁 Removed duplicate chunk #{idx}", 2)
                continue

            seen.add(h)
            out.append(c)

        if self.logger:
            self.logger.do_log(f"[DEDUP] ✅ Unique chunks kept: {len(out)} / {len(chunks)}", 1)

        return out
