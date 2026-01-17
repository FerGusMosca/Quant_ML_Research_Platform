# orchestrator/mcp_ingest_client.py
import asyncio, json, websockets
from typing import AsyncGenerator

class RAGIngestMCPClient:
    def __init__(self, mode, source, dest_root, chunk_name,
                 embedding_model, clustering_model, log_posfix, uri):
        self._mode = mode; self._source = source; self._dest_root = dest_root
        self._chunk_name = chunk_name; self._embedding_model = embedding_model
        self._clustering_model = clustering_model; self._log_posfix = log_posfix
        self._uri = uri
        self.last_output_folder = None; self.ingest_error = False; self.last_error = None

    async def execute_and_stream(self) -> AsyncGenerator[str, None]:
        payload = {"jsonrpc":"2.0","id":2,"method":"tools/call",
                   "params":{"name":"run_rag_ingest",
                             "arguments":{"mode":self._mode,"source":self._source,
                                          "dest_root":self._dest_root,
                                          "chunk_name":self._chunk_name,
                                          "embedding_model":self._embedding_model,
                                          "clustering_model":self._clustering_model,
                                          "log_posfix":self._log_posfix}}}
        try:
            async with websockets.connect(self._uri, ping_interval=None) as ws:
                await ws.send(json.dumps(payload))
                while True:
                    msg = await asyncio.wait_for(ws.recv(), timeout=120.0)
                    yield f"MSG >>> {msg}\n\n"
                    if "INGESTION COMPLETED - out_folder=" in msg:
                        self.last_output_folder = msg.split("out_folder=",1)[-1].strip().replace("\\","/")
                        return
        except Exception as e:
            self.ingest_error = True; self.last_error = str(e)