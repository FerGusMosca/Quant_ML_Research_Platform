# metadata_inventory_builder.py
import os
import json
import pandas as pd


class MetadataInventoryBuilder:

    def __init__(self, folder, logger):
        self.folder = folder
        self.logger = logger
        os.makedirs(folder, exist_ok=True)

    def save(self, items):
        json_path = os.path.join(self.folder, "corpus_inventory.json")
        csv_path = os.path.join(self.folder, "corpus_inventory.csv")

        with open(json_path, "w") as f:
            json.dump(items, f, indent=2)

        df = pd.DataFrame(items)
        df.to_csv(
            csv_path,
            index=False,
            encoding="utf-8",
            escapechar="\\"
        )

        self.logger.do_log(f"[CORPUS] Saved inventory → {json_path}", 1)
        self.logger.do_log(f"[CORPUS] Saved CSV → {csv_path}", 1)
