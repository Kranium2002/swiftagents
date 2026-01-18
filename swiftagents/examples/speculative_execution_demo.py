import asyncio

from swiftagents.core import AgentConfig, AgentRuntime, MockModelClient, ToolRegistry, ToolSpec
from swiftagents.core.models import ModelResponse


async def fast_tool(name: str, delay_s: float, query: str) -> dict:
    await asyncio.sleep(delay_s)
    return {"from": name}


async def main() -> None:
    client = MockModelClient()
    client.queue_response(
        ModelResponse(
            text="TOOL=VECTOR",
            tokens=["TOOL=VECTOR"],
            token_logprobs=[-0.6],
            top_logprobs=[{"VECTOR": -0.6, "WEB": -0.65, "NONE": -1.4}],
            usage={"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
            raw={"mock": True},
        )
    )
    client.queue_text("Answer from VECTOR")
    client.queue_text("Answer from WEB")

    tools = ToolRegistry()
    async def vector_tool(query: str) -> dict:
        return await fast_tool("VECTOR", 0.01, query)

    async def web_tool(query: str) -> dict:
        return await fast_tool("WEB", 0.02, query)

    tools.register_function(
        vector_tool,
        ToolSpec(
            name="VECTOR",
            description="Vector tool",
            input_schema={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
            example_calls=[],
            cost_hint="cheap",
            latency_hint_ms=50,
            side_effects=False,
            cacheable=False,
            cancellable=True,
        ),
    )
    tools.register_function(
        web_tool,
        ToolSpec(
            name="WEB",
            description="Web tool",
            input_schema={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
            example_calls=[],
            cost_hint="cheap",
            latency_hint_ms=50,
            side_effects=False,
            cacheable=False,
            cancellable=True,
        ),
    )

    runtime = AgentRuntime(
        client=client,
        tools=tools,
        config=AgentConfig(margin_threshold=0.2, entropy_threshold=0.0, allow_direct_answer_draft=False),
    )

    result = await runtime.run("Speculative query")
    print("Used tool:", result.used_tool)
    print("Answer:", result.answer)
    print("Tool calls:", result.metrics.tool_calls)


if __name__ == "__main__":
    asyncio.run(main())
