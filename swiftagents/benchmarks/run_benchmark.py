import asyncio
import time

from swiftagents.core import AgentConfig, AgentRuntime, MockModelClient, ToolRegistry, ToolSpec
from swiftagents.core.models import ModelResponse


async def slow_tool(name: str, delay_s: float, query: str) -> dict:
    await asyncio.sleep(delay_s)
    return {"tool": name}


def make_spec(name: str, delay_s: float) -> ToolSpec:
    return ToolSpec(
        name=name,
        description=f"{name} tool",
        input_schema={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        example_calls=[],
        cost_hint="cheap",
        latency_hint_ms=int(delay_s * 1000),
        side_effects=False,
        cacheable=False,
        cancellable=True,
    )


async def baseline_run(query: str, tools, client: MockModelClient) -> dict:
    start = time.perf_counter()
    tool_results = []
    for tool in tools:
        tool_results.append(await tool(query=query))
    client.queue_text("Baseline answer")
    answer = (await client.complete(
        messages=[{"role": "user", "content": query}],
        max_tokens=64,
        temperature=0.0,
        logprobs=False,
        top_logprobs=0,
    )).text
    latency_ms = (time.perf_counter() - start) * 1000.0
    return {
        "latency_ms": latency_ms,
        "tool_calls": len(tool_results),
        "tokens": len(answer.split()),
    }


async def swiftagents_run(query: str) -> dict:
    client = MockModelClient()
    client.queue_response(
        ModelResponse(
            text="TOOL=WEB",
            tokens=["TOOL=WEB"],
            token_logprobs=[-0.1],
            top_logprobs=[{"WEB": -0.1, "VECTOR": -1.0, "NONE": -2.0}],
            usage={"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
            raw={"mock": True},
        )
    )
    client.queue_text("Swiftagents answer")

    tools = ToolRegistry()

    async def web_tool(query: str) -> dict:
        return await slow_tool("WEB", 0.01, query)

    async def vector_tool(query: str) -> dict:
        return await slow_tool("VECTOR", 0.02, query)

    tools.register_function(web_tool, make_spec("WEB", 0.01))
    tools.register_function(vector_tool, make_spec("VECTOR", 0.02))

    runtime = AgentRuntime(client=client, tools=tools, config=AgentConfig(allow_direct_answer_draft=False))

    start = time.perf_counter()
    result = await runtime.run(query)
    latency_ms = (time.perf_counter() - start) * 1000.0
    return {
        "latency_ms": latency_ms,
        "tool_calls": result.metrics.tool_calls,
        "tokens": result.metrics.total_tokens,
    }


async def main() -> None:
    queries = ["query one", "query two", "query three"]

    async def web_tool(query: str) -> dict:
        return await slow_tool("WEB", 0.01, query)

    async def vector_tool(query: str) -> dict:
        return await slow_tool("VECTOR", 0.02, query)

    tools = [web_tool, vector_tool]
    baseline_client = MockModelClient()

    baseline_latencies = []
    swift_latencies = []

    for query in queries:
        base = await baseline_run(query, tools, baseline_client)
        swift = await swiftagents_run(query)
        baseline_latencies.append(base["latency_ms"])
        swift_latencies.append(swift["latency_ms"])

    print("Baseline avg latency (ms):", sum(baseline_latencies) / len(baseline_latencies))
    print("Swiftagents avg latency (ms):", sum(swift_latencies) / len(swift_latencies))


if __name__ == "__main__":
    asyncio.run(main())
