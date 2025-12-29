# service_layer/mcp/dispatcher.py
import json
import asyncio

from common.dto.mcp.protocol import JsonRpcResponse, JsonRpcError


class JsonRpcDispatcher:
    PROTOCOL_VERSION = "2024-11-05"

    def __init__(self, registry, bus):
        self.registry = registry
        self.bus = bus

    async def dispatch(self, raw: str, websocket) -> JsonRpcResponse:
        try:
            msg = json.loads(raw)
            req_id = msg.get("id")
            method = msg.get("method")
            params = msg.get("params", {})

            if method == "initialize":
                return JsonRpcResponse(
                    id=req_id,
                    result={
                        "protocolVersion": self.PROTOCOL_VERSION,
                        "capabilities": {"tools": {}},
                        "serverInfo": {
                            "name": "mcp-ws-server",
                            "version": "1.0.0"
                        }
                    }
                )

            if method == "tools/list":
                return JsonRpcResponse(
                    id=req_id,
                    result=self.registry.list_specs()
                )

            if method == "tools/call":
                tool_name = params.get("name")
                arguments = params.get("arguments", {})

                tool = self.registry.get(tool_name)

                result = tool.handler(arguments)

                job_id = result.get("job_id")
                if job_id:
                    await self.bus.subscribe(job_id, websocket)

                return JsonRpcResponse(
                    id=req_id,
                    result={
                        "content": [{
                            "type": "text",
                            "text": json.dumps(result)
                        }]
                    }
                )

            return JsonRpcResponse(
                id=req_id,
                error=JsonRpcError(-32601, "Method not found")
            )

        except Exception as e:
            return JsonRpcResponse(
                id=None,
                error=JsonRpcError(-32000, str(e))
            )
