# drift_detector.py
import os
import json


class DriftDetector:

    def __init__(self, logger):
        self.logger = logger
        self.inventory_path = "./output/corpus_metadata/corpus_hashes.json"
        self.prev = {}

        if os.path.exists(self.inventory_path):
            with open(self.inventory_path, "r") as f:
                self.prev = json.load(f)

    def apply_status(self, items):
        new_hashes = {}

        for m in items:
            key = m["path"]
            new_hashes[key] = m["sha256_file"]

            if key not in self.prev:
                m["status"] = "new"
            elif self.prev[key] != m["sha256_file"]:
                m["status"] = "modified"
            else:
                m["status"] = "unchanged"

        # detect deleted
        for old in self.prev.keys():
            if old not in new_hashes:
                items.append({
                    "path": old,
                    "status": "deleted"
                })

        with open(self.inventory_path, "w") as f:
            json.dump(new_hashes, f, indent=2)

        return items
