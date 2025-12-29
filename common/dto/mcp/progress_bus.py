# service_layer/mcp/progress_bus.py
import asyncio
import json
from typing import Dict, Set


class ProgressBus:
    """
    In-memory async pub/sub bus.
    One job_id -> many WebSocket subscribers.
    Used ONLY to stream progress events to clients.
    """

    def __init__(self):
        self._subscribers: Dict[str, Set] = {}
        self._lock = asyncio.Lock()

    async def subscribe(self, job_id: str, websocket) -> None:
        """
        Attach a websocket to a job_id.
        """
        async with self._lock:
            self._subscribers.setdefault(job_id, set()).add(websocket)

    async def unsubscribe(self, job_id: str, websocket) -> None:
        """
        Detach a websocket from a job_id.
        """
        async with self._lock:
            if job_id in self._subscribers:
                self._subscribers[job_id].discard(websocket)
                if not self._subscribers[job_id]:
                    del self._subscribers[job_id]

    async def unsubscribe_all(self, websocket):
        async with self._lock:
            for job_id in list(self._subscribers.keys()):
                self._subscribers[job_id].discard(websocket)
                if not self._subscribers[job_id]:
                    del self._subscribers[job_id]


    async def publish(self, job_id: str, payload: dict) -> None:
        """
        Send a progress event to all subscribers of the job.
        """
        async with self._lock:
            targets = list(self._subscribers.get(job_id, []))

        if not targets:
            return

        message = json.dumps({
            "jsonrpc": "2.0",
            "method": "job/progress",
            "params": {
                "job_id": job_id,
                **payload
            }
        })

        for ws in targets:
            try:
                await ws.send(message)
            except Exception:
                # Client is dead → ignore, cleanup happens on unsubscribe
                pass
