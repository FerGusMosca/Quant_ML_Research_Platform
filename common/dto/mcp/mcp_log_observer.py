"""
MCP Log Observer with Langfuse Integration
==========================================
Publishes logs to both the client (via bus) and Langfuse (via ObsHttpClient).
"""

import asyncio
from typing import Optional

from common.util.observability.langfuse.observability_context import ObsContext
from common.util.observability.obs_client import ObsHttpClient
from common.util.std_in_out.ml_settings_loader import MLSettingsLoader
from framework.common.logger.message_type import MessageType


# Lazy import to avoid circular dependencies
_obs_client = None


def _get_obs_client():
    """Lazy initialization of ObsHttpClient."""
    global _obs_client
    if _obs_client is None:
        try:


            loader = MLSettingsLoader()
            config = loader.load_settings("./configs/commands_mgr.ini")

            proxy_url = config.get("OBSERVABILITY_PROXY_URL", "http://localhost:7003")

            _obs_client = ObsHttpClient(
                service_id=None,  # Will be set per-request from context
                proxy_url=proxy_url
            )
        except Exception as e:
            print(f"[OBS] Failed to initialize ObsHttpClient: {e}")
            _obs_client = False  # Mark as failed, don't retry

    return _obs_client if _obs_client else None


class MCPLogObserver:
    """
    Observer that publishes logs to:
    1. Client via bus (existing behavior)
    2. Langfuse via ObsHttpClient (new)
    """

    def __init__(self, bus):
        self.bus = bus
        self.loop = asyncio.get_running_loop()

    def on_log(self, msg: str, level: MessageType, job_id: str):
        # Original behavior: publish to client
        self.loop.call_soon_threadsafe(
            asyncio.create_task,
            self.bus.publish(job_id, {
                "level": level.name,
                "message": msg
            })
        )

        # New: send to Langfuse (non-blocking)
        self._send_to_langfuse(msg, level, job_id)

    def _send_to_langfuse(self, msg: str, level: MessageType, job_id: str):
        """Send log to Langfuse via observability proxy."""
        try:
            client = _get_obs_client()
            if not client:
                return

            # Get context for this job
            ctx = ObsContext.get(job_id)

            # Build payload
            service_id = ctx.service_id if ctx else "unknown"
            operation = ctx.operation_name if ctx else "unknown"

            # Map MessageType to Langfuse level
            level_map = {
                MessageType.INFO: "DEFAULT",
                MessageType.DEBUG: "DEBUG",
                MessageType.WARNING: "WARNING",
                MessageType.ERROR: "ERROR"
            }

            payload = {
                "service_id": service_id,
                "node_name": f"{operation}_log",
                "input": {"message": msg},
                "output": {"logged": True},
                "level": level_map.get(level, "DEFAULT"),
                "trace_id": job_id,
                "trace_name": operation,
                "metadata": ctx.metadata if ctx else {}
            }

            # Fire and forget (don't block the log)
            self.loop.run_in_executor(None, lambda: client._send("/log", payload))

        except Exception as e:
            # Never let observability break the main flow
            pass