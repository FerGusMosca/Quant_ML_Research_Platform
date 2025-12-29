from dataclasses import dataclass
from typing import Any, Dict, Callable

JSON = Dict[str, Any]

@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: JSON

class Tool:
    def __init__(self, spec: ToolSpec, handler: Callable[[JSON], Any]):
        self.spec = spec
        self.handler = handler

class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.spec.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.spec.name}")
        self._tools[tool.spec.name] = tool

    def list_specs(self) -> JSON:
        return {
            "tools": [{
                "name": t.spec.name,
                "description": t.spec.description,
                "inputSchema": t.spec.input_schema
            } for t in self._tools.values()]
        }

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise KeyError(f"Unknown tool: {name}")
        return self._tools[name]
