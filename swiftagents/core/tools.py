from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Protocol, Union

try:
    from pydantic import BaseModel
except ImportError:  # pragma: no cover
    BaseModel = None  # type: ignore


@dataclass
class ToolResult:
    ok: bool
    data: Any
    error: Optional[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolSpec:
    name: str
    description: str
    input_schema: Union[Dict[str, Any], Any]
    example_calls: List[Dict[str, Any]]
    cost_hint: str
    latency_hint_ms: int
    side_effects: bool
    cacheable: bool
    cancellable: bool


class Tool(Protocol):
    spec: ToolSpec

    async def __call__(self, **kwargs: Any) -> ToolResult:
        ...


@dataclass
class RegisteredTool:
    spec: ToolSpec
    handler: Callable[..., Awaitable[ToolResult]]


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: Dict[str, RegisteredTool] = {}

    def register(self, tool: Tool) -> None:
        if not hasattr(tool, "spec"):
            raise ValueError("Tool missing spec attribute")
        spec = tool.spec
        if spec.name in self._tools:
            raise ValueError(f"Tool name already registered: {spec.name}")
        if inspect.iscoroutinefunction(tool):
            handler = self._wrap_callable(tool, allow_sync=False)
        elif inspect.iscoroutinefunction(tool.__call__):
            handler = self._wrap_callable(tool.__call__, allow_sync=False)
        else:
            raise ValueError(f"Tool {spec.name} must be async")
        self._tools[spec.name] = RegisteredTool(spec=spec, handler=handler)

    def register_function(
        self,
        func: Callable[..., Any],
        spec: ToolSpec,
        *,
        allow_sync: bool = True,
    ) -> None:
        if spec.name in self._tools:
            raise ValueError(f"Tool name already registered: {spec.name}")
        handler = self._wrap_callable(func, allow_sync=allow_sync)
        self._tools[spec.name] = RegisteredTool(spec=spec, handler=handler)

    def get(self, name: str) -> RegisteredTool:
        if name not in self._tools:
            raise KeyError(f"Unknown tool: {name}")
        return self._tools[name]

    def list_specs(self) -> List[ToolSpec]:
        return [tool.spec for tool in self._tools.values()]

    def list_tool_names(self) -> List[str]:
        return list(self._tools.keys())

    def validate_args(self, spec: ToolSpec, args: Dict[str, Any]) -> None:
        schema = spec.input_schema
        if BaseModel is not None and isinstance(schema, type) and issubclass(schema, BaseModel):
            schema(**args)
            return
        if isinstance(schema, dict):
            required = schema.get("required") or []
            for key in required:
                if key not in args:
                    raise ValueError(f"Missing required arg: {key}")

    def _wrap_callable(self, func: Callable[..., Any], *, allow_sync: bool) -> Callable[..., Awaitable[ToolResult]]:
        if inspect.iscoroutinefunction(func):
            async def _handler(**kwargs: Any) -> ToolResult:
                result = await func(**kwargs)
                return _normalize_tool_result(result)
        else:
            if not allow_sync:
                raise ValueError("Sync tool functions not allowed")

            async def _handler(**kwargs: Any) -> ToolResult:
                result = await asyncio.to_thread(func, **kwargs)
                return _normalize_tool_result(result)

        return _handler


def _normalize_tool_result(result: Any) -> ToolResult:
    if isinstance(result, ToolResult):
        return result
    return ToolResult(ok=True, data=result, error=None, metadata={})


def tool(spec: ToolSpec) -> Callable[[Callable[..., Awaitable[ToolResult]]], Any]:
    def decorator(func: Callable[..., Awaitable[ToolResult]]) -> Any:
        setattr(func, "spec", spec)
        return func

    return decorator
