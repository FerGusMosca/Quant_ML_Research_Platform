# service_layer/mcp/mcp_log_observer.py
import asyncio
from framework.common.logger.message_type import MessageType

class MCPLogObserver:
    def __init__(self, bus):
        self.bus = bus
        self.loop = asyncio.get_running_loop()

    def on_log(self, msg, level: MessageType,job_id):
        self.loop.call_soon_threadsafe(
            asyncio.create_task,
            self.bus.publish(job_id, {
                "level": level.name,
                "message": msg
            })
        )
