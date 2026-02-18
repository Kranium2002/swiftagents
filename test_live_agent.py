import asyncio
import os
from dotenv import load_dotenv
from swiftagents.core import (
    AgentConfig,
    AgentRuntime,
    OpenAIChatCompletionsClient,
    ToolRegistry,
    ToolSpec,
)

load_dotenv()

def get_weather(query: str) -> dict:
    """Fake weather lookup."""
    return {"location": "San Francisco", "temp_f": 62, "condition": "Foggy"}

def do_math(query: str) -> dict:
    """Fake math solver."""
    return {"expression": query, "result": 42}

def tell_joke(query: str) -> dict:
    """Fake joke teller."""
    return {"joke": "Why do programmers prefer dark mode? Because light attracts bugs."}

async def main() -> None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not found in .env")
        return

    client = OpenAIChatCompletionsClient(api_key=api_key, model="gpt-4o-mini")

    tools = ToolRegistry()
    tools.register_function(
        get_weather,
        ToolSpec(
            name="WEATHER",
            description="Get current weather for a location",
            input_schema={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
            example_calls=[],
            cost_hint="low",
            latency_hint_ms=100,
            side_effects=False,
            cacheable=True,
            cancellable=True,
        ),
    )
    tools.register_function(
        do_math,
        ToolSpec(
            name="MATH",
            description="Solve math problems and calculations",
            input_schema={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
            example_calls=[],
            cost_hint="low",
            latency_hint_ms=50,
            side_effects=False,
            cacheable=True,
            cancellable=True,
        ),
    )
    tools.register_function(
        tell_joke,
        ToolSpec(
            name="JOKE",
            description="Tell a funny joke or humor",
            input_schema={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
            example_calls=[],
            cost_hint="low",
            latency_hint_ms=50,
            side_effects=False,
            cacheable=True,
            cancellable=True,
        ),
    )

    config = AgentConfig(answer_max_tokens=256)
    runtime = AgentRuntime(client=client, tools=tools, config=config)

    queries = [
        "What's the weather like in San Francisco?",
        "What is 15 * 27?",
        "Tell me a funny programming joke",
    ]

    for query in queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print('='*60)
        try:
            result = await runtime.run(query)
            print(f"Tool used: {result.used_tool}")
            print(f"Answer: {result.answer}")
            print(f"Metrics: tokens={result.metrics.total_tokens}, latency={result.metrics.latency_ms:.0f}ms")
        except Exception as e:
            print(f"ERROR: {type(e).__name__}: {e}")

    await client.close()
    print("\nAll queries completed successfully!")

if __name__ == "__main__":
    asyncio.run(main())
