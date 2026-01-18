# swiftagents

Superfast, logprob-native, async-first agent runtime.

## Why swiftagents

- Logprob-native routing and uncertainty
- Tool-agnostic, model-agnostic (strict about logprobs)
- Async-first with bounded speculation (max 2 tools)
- Cost-aware, cacheable, and observable
- Optional judge pipeline

## Install

```bash
pip install swiftagents
```

Local development:

```bash
pip install -e .[dev]
```

## Quickstart

```python
import asyncio

from swiftagents.core import AgentRuntime, AgentConfig, MockModelClient, ToolRegistry, ToolSpec


def web_search(query: str) -> dict:
    # Put any code here: Pinecone, DB, APIs, etc.
    return {"snippet": "Example result"}


async def main():
    client = MockModelClient()
    client.queue_text("TOOL=WEB")
    client.queue_text("Answer using web evidence")

    spec = ToolSpec(
        name="WEB",
        description="Search the web",
        input_schema={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        example_calls=[],
        cost_hint="medium",
        latency_hint_ms=200,
        side_effects=False,
        cacheable=True,
        cancellable=True,
    )

    tools = ToolRegistry()
    tools.register_function(web_search, spec)

    runtime = AgentRuntime(client=client, tools=tools, config=AgentConfig())
    result = await runtime.run("Find the latest overview")
    print(result.answer)


asyncio.run(main())
```

## Core concepts

### Model clients (logprobs required)

`swiftagents` requires token-level logprobs for routing. If a backend cannot provide them, it hard-errors.

Supported clients:
- `OpenAIChatCompletionsClient`
- `VLLMOpenAICompatibleClient`
- `MockModelClient` (tests and examples)

### Tools

Tools are async callables with a `ToolSpec` and return `ToolResult`.

```python
from swiftagents.core import ToolSpec, ToolResult

class MyTool:
    def __init__(self):
        self.spec = ToolSpec(
            name="RAG",
            description="Retrieve from docs",
            input_schema={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
            example_calls=[],
            cost_hint="medium",
            latency_hint_ms=200,
            side_effects=False,
            cacheable=True,
            cancellable=True,
        )

    async def __call__(self, **kwargs):
        return ToolResult(ok=True, data={"docs": []}, error=None, metadata={})
```

### Register functions directly (no tool classes)

Use any code inside a function (sync or async) and register it with a `ToolSpec`.

```python
from swiftagents.core import ToolRegistry, ToolSpec, ToolResult

def pinecone_search(query: str) -> dict:
    # Put any code here (Pinecone, DB, API, etc.)
    # return raw data or ToolResult
    return {"matches": [{"id": "doc1", "score": 0.92}]}

spec = ToolSpec(
    name="PINECONE",
    description="Vector search over Pinecone",
    input_schema={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
    example_calls=[],
    cost_hint="medium",
    latency_hint_ms=200,
    side_effects=False,
    cacheable=True,
    cancellable=True,
)

registry = ToolRegistry()
registry.register_function(pinecone_search, spec)
```

If you prefer decorators:

```python
from swiftagents.core import ToolRegistry, ToolSpec, tool

spec = ToolSpec(
    name="PINECONE",
    description="Vector search over Pinecone",
    input_schema={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
    example_calls=[],
    cost_hint="medium",
    latency_hint_ms=200,
    side_effects=False,
    cacheable=True,
    cancellable=True,
)

@tool(spec)
async def pinecone_tool(query: str):
    return {"matches": [{"id": "doc1", "score": 0.92}]}

registry = ToolRegistry()
registry.register(pinecone_tool)
```

### Routing (logprob-gated)

Routing prompts the same LLM to output `TOOL=<LABEL>` where `LABEL` is `NONE` or a shortlist tool name.
Confidence is computed using logprobs, entropy, and margin. Low confidence triggers bounded speculation.

### Multi-tool routing modes

`AgentConfig.multi_tool_mode` controls how the runtime selects multiple tools:
- `single`: default single-label routing (bounded speculation when uncertain).
- `multi_label`: pick multiple tools from one router call using logprob thresholds.
- `multi_intent`: lightweight heuristic splitting, then route each segment.
- `decompose`: logprob-gated split decision + LLM decomposition into sub-questions, then route each.

All multi-tool modes merge tool evidence and produce one final answer.

### Judge

`Judge` behaves like a tool. It can be disabled, run a cheap LLM, and optionally escalate to a stronger LLM.
It can also run deterministic stage0 checks.

### Caching and observability

- Tool and model decision caches with TTL
- Structured trace events
- Token usage metrics and wasted work ratio

## Examples

```bash
python -m swiftagents.examples.tool_selection
python -m swiftagents.examples.speculative_execution_demo
python -m swiftagents.examples.function_tool_demo
```

## Benchmarks

```bash
python -m swiftagents.benchmarks.run_benchmark
```

## Tests

```bash
pytest
```

## Design notes

- The router is logprob-native; labels should be compact and stable (prefer short uppercase names).
- Speculation is bounded to two tools and never speculative for side-effecting tools unless explicitly allowed.
- All runtime stages are async-first.
