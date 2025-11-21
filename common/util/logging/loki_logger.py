# ALL COMMENTS IN ENGLISH
import logging
import requests
import json
import time

class LokiLogger:
    """
    Simple Loki HTTP client for pushing logs to Grafana Loki.
    """

    def __init__(self, loki_url, app_name="ml_research"):
        self.loki_url = loki_url.rstrip("/") + "/loki/api/v1/push"
        self.app_name = app_name

    def push(self, level, msg):
        ts = int(time.time() * 1e9)   # nanoseconds timestamp

        payload = {
            "streams": [
                {
                    "stream": {
                        "app": self.app_name,
                        "level": level
                    },
                    "values": [
                        [str(ts), msg]
                    ]
                }
            ]
        }

        try:
            requests.post(
                self.loki_url,
                data=json.dumps(payload),
                headers={"Content-Type": "application/json"},
                timeout=2
            )
        except Exception as e:
            print(f"[LokiLogger] Failed to push log: {e}")
