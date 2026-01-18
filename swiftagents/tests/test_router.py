import pytest

from swiftagents.core.models import MockModelClient, ModelResponse
from swiftagents.core.router import RouterConfig, ToolRouter, shortlist_tools
from swiftagents.core.tools import ToolSpec


def _tool_spec(name: str, description: str) -> ToolSpec:
    return ToolSpec(
        name=name,
        description=description,
        input_schema={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        example_calls=[],
        cost_hint="cheap",
        latency_hint_ms=10,
        side_effects=False,
        cacheable=True,
        cancellable=True,
    )


def test_shortlist_prefers_relevant_tools():
    tools = [
        _tool_spec("WEB", "Search the web for current events"),
        _tool_spec("RAG", "Retrieve from local documents"),
        _tool_spec("CODE", "Execute code snippets"),
    ]
    shortlist = shortlist_tools("search the web for latest news", tools, max_tools=2)
    assert shortlist[0].name == "WEB"


@pytest.mark.asyncio
async def test_router_confident_tool_choice():
    client = MockModelClient()
    response = ModelResponse(
        text="TOOL=WEB",
        tokens=["TOOL=WEB"],
        token_logprobs=[-0.05],
        top_logprobs=[{"WEB": -0.05, "RAG": -1.2, "NONE": -2.0}],
        usage={"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
        raw={"mock": True},
    )
    client.queue_response(response)
    router = ToolRouter(client, RouterConfig())

    decision = await router.decide_tool("Find news", [_tool_spec("WEB", ""), _tool_spec("RAG", "")])
    assert decision.label == "WEB"
    assert decision.margin > 0.4
