# orchestrator/mcp_ingest_client.py
import asyncio
import json
import websockets
from typing import Optional, AsyncGenerator


class RAGIngestMCPClient:
    """
    MCP client responsible for executing RAG ingestion and consuming job/progress events.

    Contract:
    - FINISHED ONLY when explicit terminal event is received.
    - Streaming raw messages yielded.
    - Timeout without completion => failure.
    - Any exception => failure.
    """

    def __init__(
        self,
        uri: str,
        mode: str,
        source: str,
        dest_root: str,
        chunk_name: str,
        embedding_model: Optional[str] = None,
        clustering_model: Optional[str] = None,
        log_posfix: Optional[str] = None,
    ):
        self.uri = uri
        self.mode = mode
        self.source = source
        self.dest_root = dest_root
        self.chunk_name = chunk_name
        self.embedding_model = embedding_model
        self.clustering_model = clustering_model
        self.log_posfix = log_posfix

        # Execution state
        self.success: bool = False
        self.last_error: Optional[str] = None
        self.last_output_folder: Optional[str] = None

    async def execute_and_stream(self) -> AsyncGenerator[str, None]:
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "run_rag_ingest",
                "arguments": {
                    "mode": self.mode,
                    "source": self.source,
                    "dest_root": self.dest_root,
                    "chunk_name": self.chunk_name,
                    "embedding_model": self.embedding_model,
                    "clustering_model": self.clustering_model,
                    "log_posfix": self.log_posfix,
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
                        self.last_error = "Timeout waiting for MCP ingestion completion event"
                        self.success = False
                        return

                    # Stream raw messages to caller
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

                    # MCP embeds structured events as JSON strings
                    if not message.lstrip().startswith("{"):
                        continue

                    try:
                        inner = json.loads(message)
                    except Exception:
                        continue

                    # ----------- TERMINAL EVENT (CRITICAL) -----------
                    if inner.get("event") == "completed":
                        self.success = True
                        self.last_output_folder = (
                            inner.get("out_folder")
                            or inner.get("output_folder")
                        )
                        return

        except Exception as e:
            self.last_error = str(e)
            self.success = False
