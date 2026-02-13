import requests
import uuid

class ObsClient:
    """
    Bridge client to send logs to the proxy on port 7003.
    """
    def __init__(self, port=7003):
        self.proxy_url = f"http://localhost:{port}/log"
        self.trace_id = str(uuid.uuid4())

    def log_event(self, node_name, input_data, output_data, parent_id=None):
        payload = {
            "trace_id": self.trace_id,
            "node_name": node_name,
            "input": str(input_data),
            "output": str(output_data),
            "parent_id": parent_id
        }
        try:
            # Short timeout to avoid blocking main logic
            requests.post(self.proxy_url, json=payload, timeout=1)
        except Exception:
            pass # Keep ML process stable if proxy is down