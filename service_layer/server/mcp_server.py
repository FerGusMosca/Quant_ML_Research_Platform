import asyncio
import uuid

import websockets
from websockets import ConnectionClosed

from common.dto.mcp.dispatcher import JsonRpcDispatcher
from framework.common.logger.message_type import MessageType


class MCPServer:
    def __init__(self, host: str, port: int, dispatcher: JsonRpcDispatcher,bus, logger):
        self.host = host
        self.port = port
        self.dispatcher = dispatcher
        self.bus=bus
        self.logger = logger

    async def start(self) -> None:
        async with websockets.serve(self._handle_client, self.host, self.port):
            self.logger.do_log(
                f"[MCP] 🚀 MCP WS listening on ws://{self.host}:{self.port}",
                MessageType.INFO
            )
            await asyncio.Future()  # run forever

    async def _handle_client(self, websocket):
        try:
            async for raw in websocket:
                response = await self.dispatcher.dispatch(raw, websocket)
                await websocket.send(response.to_json())
        except ConnectionClosed:
            pass
        except Exception as e:
            self.logger.do_log(f"[MCP] WS error: {e}", MessageType.ERROR)
        finally:
            await self.dispatcher.bus.unsubscribe_all(websocket)
