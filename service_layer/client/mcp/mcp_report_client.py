import asyncio
import json
import websockets
from typing import Optional


class ReportMCPClient:
    """
    MCP client responsible for executing a report and consuming job/progress events.

    Contract:
    - The job is considered FINISHED ONLY when an explicit
      {"event": "completed"} message is received.
    - Streaming logs are yielded to the caller.
    - Timeout without completion => failure.
    - Any exception => failure.

    This client encapsulates ALL MCP protocol knowledge.
    """

    def __init__(self, uri: str, report: str, arguments: dict):
        self.uri = uri
        self.report = report
        self.arguments = arguments

        # Execution state
        self.success: bool = False
        self.last_error: Optional[str] = None

        # Parsed completion payload
        self.summary: Optional[dict] = None
        self.completed_report: Optional[str] = None

    async def execute_and_stream(self):
        """
        Executes the MCP report and yields raw websocket messages as they arrive.
        Terminates ONLY on explicit 'completed' event.
        """

        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "run_report",
                "arguments": {
                    "report": self.report,
                    **self.arguments,
                },
            },
        }

        try:
            async with websockets.connect(self.uri, ping_interval=None) as ws:
                await ws.send(json.dumps(payload))

                while True:
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=120)
                    except asyncio.TimeoutError:
                        # Timeout without completion is a FAILURE
                        self.last_error = "Timeout waiting for MCP completion event"
                        self.success = False
                        return

                    # Always stream raw messages to caller
                    yield raw

                    # Parse outer JSON-RPC envelope
                    try:
                        outer = json.loads(raw)
                    except Exception:
                        continue

                    if outer.get("method") != "job/progress":
                        continue

                    message = outer.get("params", {}).get("message", "")
                    if not message:
                        continue

                    # The MCP protocol embeds structured events as JSON strings
                    if not message.lstrip().startswith("{"):
                        continue

                    try:
                        inner = json.loads(message)
                    except Exception:
                        continue

                    # ----------- TERMINAL EVENT (CRITICAL) -----------
                    if inner.get("event") == "completed":
                        self.success = True
                        self.summary = inner.get("summary")
                        self.completed_report = inner.get("report")
                        return

        except Exception as e:
            self.last_error = str(e)
            self.success = False