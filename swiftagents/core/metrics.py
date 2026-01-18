from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TraceEvent:
    name: str
    timestamp: float
    duration_ms: Optional[float]
    data: Dict[str, Any]


class Trace:
    def __init__(self) -> None:
        self.events: List[TraceEvent] = []

    def add(self, name: str, *, data: Optional[Dict[str, Any]] = None, duration_ms: Optional[float] = None) -> None:
        self.events.append(
            TraceEvent(
                name=name,
                timestamp=time.time(),
                duration_ms=duration_ms,
                data=data or {},
            )
        )


@dataclass
class Metrics:
    tool_calls: int = 0
    tool_calls_used: int = 0
    tool_calls_wasted: int = 0
    model_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0

    def add_usage(self, usage: Dict[str, int]) -> None:
        self.prompt_tokens += usage.get("prompt_tokens", 0)
        self.completion_tokens += usage.get("completion_tokens", 0)
        self.total_tokens += usage.get("total_tokens", 0)

    @property
    def wasted_work_ratio(self) -> float:
        if self.tool_calls == 0:
            return 0.0
        return self.tool_calls_wasted / max(self.tool_calls, 1)


@dataclass
class CostTracker:
    tool_costs: Dict[str, float] = field(default_factory=dict)
    model_cost: float = 0.0
    budget: Optional[float] = None

    def add_tool_cost(self, name: str, cost: float) -> None:
        self.tool_costs[name] = self.tool_costs.get(name, 0.0) + cost

    def add_model_cost(self, cost: float) -> None:
        self.model_cost += cost

    @property
    def total_cost(self) -> float:
        return self.model_cost + sum(self.tool_costs.values())

    def budget_remaining(self) -> Optional[float]:
        if self.budget is None:
            return None
        return self.budget - self.total_cost
