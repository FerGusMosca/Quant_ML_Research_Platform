import asyncio
import websockets
from websockets import ConnectionClosed
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK

from common.dto.mcp.dispatcher import JsonRpcDispatcher
from framework.common.logger.message_type import MessageType


class MCPServer:
    """
    WebSocket-based MCP (Model Context Protocol) server.
    Handles JSON-RPC dispatch over persistent WebSocket connections.
    """

    def __init__(self, host: str, port: int, dispatcher: JsonRpcDispatcher, bus, logger):
        self.host = host
        self.port = port
        self.dispatcher = dispatcher
        self.bus = bus
        self.logger = logger

    async def start(self) -> None:
        """Start the WebSocket server and run indefinitely."""
        async with websockets.serve(self._handle_client, self.host, self.port):
            self.logger.do_log(
                f"[MCP] 🚀 MCP WS listening on ws://{self.host}:{self.port}",
                MessageType.INFO
            )
            await asyncio.Future()  # Block forever

    async def _handle_client(self, websocket):
        """
        Handle a single WebSocket client connection.

        Gracefully handles various disconnection scenarios without
        polluting logs with expected client disconnection events.
        """
        try:
            async for raw in websocket:
                response = await self.dispatcher.dispatch(raw, websocket)
                await websocket.send(response.to_json())

        except ConnectionClosedOK:
            # Client closed the connection cleanly (normal shutdown)
            # No action needed - this is expected behavior
            pass

        except ConnectionClosedError as e:
            # Client disconnected abruptly but this is still expected
            # (e.g., browser tab closed, network drop, client crash)
            # Log at DEBUG level for troubleshooting if needed
            self.logger.do_log(
                f"[MCP] Client disconnected (code={e.code})",
                MessageType.DEBUG,
                None
            )

        except ConnectionClosed:
            # Generic fallback for any other ConnectionClosed variants
            pass

        except EOFError:
            # Handles "line without CRLF" error from websockets library
            # Occurs when client terminates without proper WS close handshake
            pass

        except Exception as e:
            # Only log actual unexpected errors, not disconnection noise
            error_type = type(e).__name__
            error_msg = str(e).lower()

            # Filter out common disconnection-related errors that slip through
            is_disconnection_noise = (
                    "eof" in error_msg or
                    "handshake" in error_msg or
                    "connection" in error_msg
            )

            if not is_disconnection_noise:
                self.logger.do_log(
                    f"[MCP] Unexpected WS error ({error_type}): {e}",
                    MessageType.ERROR
                )

        finally:
            # Always clean up subscriptions when client disconnects
            await self.dispatcher.bus.unsubscribe_all(websocket)