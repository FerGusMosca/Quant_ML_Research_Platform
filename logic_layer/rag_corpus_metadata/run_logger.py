import os, json, datetime

class RunLogger:
    def __init__(self, output_folder):
        self.output_folder = output_folder
        self.log_path = os.path.join(output_folder, "run_log.txt")
        self.summary_path = os.path.join(output_folder, "run_summary.json")

    def write_log(self, msg):
        ts = datetime.datetime.utcnow().isoformat()
        with open(self.log_path, "a") as f:
            f.write(f"[{ts}] {msg}\n")

    def write_summary(self, items):
        summary = {
            "total": len(items),
            "new": sum(1 for x in items if x["status"] == "new"),
            "modified": sum(1 for x in items if x["status"] == "modified"),
            "unchanged": sum(1 for x in items if x["status"] == "unchanged"),
            "deleted": sum(1 for x in items if x["status"] == "deleted"),
            "ts": datetime.datetime.utcnow().isoformat()
        }
        with open(self.summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        return summary
