import json
import os


class JsonFileReader():



    @staticmethod
    def load_json_file(root_path, file_name):
        tag_file = os.path.join(root_path, file_name)

        if not os.path.isfile(tag_file):
            raise FileNotFoundError(f"[CONFIG] JSON file not found: {tag_file}")

        try:
            with open(tag_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"[CONFIG] Invalid JSON format in {tag_file}: {e}")
        except Exception as e:
            raise RuntimeError(f"[CONFIG] Failed loading JSON file {tag_file}: {e}")
