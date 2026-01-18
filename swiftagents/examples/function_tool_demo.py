import asyncio

from swiftagents.core import AgentConfig, AgentRuntime, MockModelClient, ToolRegistry, ToolSpec


def pinecone_search(query: str) -> dict:
    # Replace this with real Pinecone or any custom code.
    return {"matches": [{"id": "doc1", "score": 0.92, "text": "Example"}]}


async def main() -> None:
    client = MockModelClient()
    client.queue_text("TOOL=PINECONE")
    client.queue_text("Answer using Pinecone results.")

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

    tools = ToolRegistry()
    tools.register_function(pinecone_search, spec)

    runtime = AgentRuntime(client=client, tools=tools, config=AgentConfig())
    result = await runtime.run("Search vector DB")

    print("Used tool:", result.used_tool)
    print("Answer:", result.answer)


if __name__ == "__main__":
    asyncio.run(main())
