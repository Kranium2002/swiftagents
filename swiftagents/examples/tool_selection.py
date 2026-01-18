import asyncio

from swiftagents.core import AgentConfig, AgentRuntime, MockModelClient, ToolRegistry, ToolSpec


def web_search(query: str) -> dict:
    return {"snippet": "Example web result"}


def vector_search(query: str) -> dict:
    return {"matches": [{"id": "doc1", "score": 0.9}]}


async def main() -> None:
    client = MockModelClient()
    client.queue_text("TOOL=WEB")
    client.queue_text("Here is the answer using web evidence.")

    tools = ToolRegistry()
    tools.register_function(
        web_search,
        ToolSpec(
            name="WEB",
            description="Search the web",
            input_schema={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
            example_calls=[],
            cost_hint="medium",
            latency_hint_ms=200,
            side_effects=False,
            cacheable=True,
            cancellable=True,
        ),
    )
    tools.register_function(
        vector_search,
        ToolSpec(
            name="VECTOR",
            description="Vector search over a database",
            input_schema={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
            example_calls=[],
            cost_hint="medium",
            latency_hint_ms=150,
            side_effects=False,
            cacheable=True,
            cancellable=True,
        ),
    )

    runtime = AgentRuntime(client=client, tools=tools, config=AgentConfig())
    result = await runtime.run("Find the latest SwiftAgents overview")

    print("Used tool:", result.used_tool)
    print("Answer:", result.answer)


if __name__ == "__main__":
    asyncio.run(main())
