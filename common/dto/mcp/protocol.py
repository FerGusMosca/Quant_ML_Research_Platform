import json
from dataclasses import dataclass
from typing import Any, Dict, Optional

JSON = Dict[str, Any]

@dataclass(frozen=True)
class JsonRpcError:
    code: int
    message: str

@dataclass
class JsonRpcResponse:
    id: Optional[int]
    result: Optional[JSON] = None
    error: Optional[JsonRpcError] = None

    def to_json(self) -> str:
        payload: JSON = {"jsonrpc": "2.0", "id": self.id}
        if self.error:
            payload["error"] = {"code": self.error.code, "message": self.error.message}
        else:
            payload["result"] = self.result
        return json.dumps(payload)
