import asyncio

import pytest

from swiftagents.core.judge import Judge, JudgeConfig
from swiftagents.core.models import MockModelClient, ModelResponse
from swiftagents.core.scheduler import AgentConfig, AgentRuntime
from swiftagents.core.tools import ToolRegistry, ToolResult, ToolSpec


class DummyTool:
    def __init__(self, name: str, delay_s: float = 0.0) -> None:
        self.spec = ToolSpec(
            name=name,
            description=f"{name} tool",
            input_schema={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
            example_calls=[],
            cost_hint="cheap",
            latency_hint_ms=5,
            side_effects=False,
            cacheable=False,
            cancellable=True,
        )
        self._delay_s = delay_s

    async def __call__(self, **kwargs):
        await asyncio.sleep(self._delay_s)
        return ToolResult(ok=True, data={"source": self.spec.name}, error=None, metadata={})


@pytest.mark.asyncio
async def test_runtime_speculates_two_tools():
    client = MockModelClient()
    client.queue_response(
        ModelResponse(
            text="TOOL=RAG",
            tokens=["TOOL=RAG"],
            token_logprobs=[-0.5],
            top_logprobs=[{"RAG": -0.5, "WEB": -0.55, "NONE": -1.5}],
            usage={"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
            raw={"mock": True},
        )
    )
    client.queue_text("Answer from RAG")
    client.queue_text("Answer from WEB")

    tools = ToolRegistry()
    tools.register(DummyTool("RAG", delay_s=0.01))
    tools.register(DummyTool("WEB", delay_s=0.02))

    runtime = AgentRuntime(
        client=client,
        tools=tools,
        config=AgentConfig(margin_threshold=0.2, entropy_threshold=0.0, allow_direct_answer_draft=False),
    )

    result = await runtime.run("Need info")
    assert result.metrics.tool_calls == 2
    assert result.used_tool in {"RAG", "WEB"}


@pytest.mark.asyncio
async def test_tool_timeout_falls_back_to_direct_answer():
    client = MockModelClient()
    client.queue_response(
        ModelResponse(
            text="TOOL=WEB",
            tokens=["TOOL=WEB"],
            token_logprobs=[-0.1],
            top_logprobs=[{"WEB": -0.1, "NONE": -2.0}],
            usage={"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
            raw={"mock": True},
        )
    )
    client.queue_text("Direct answer")

    tools = ToolRegistry()
    tools.register(DummyTool("WEB", delay_s=0.05))

    runtime = AgentRuntime(
        client=client,
        tools=tools,
        config=AgentConfig(tool_timeout_s=0.01, allow_direct_answer_draft=False),
    )

    result = await runtime.run("Need info")
    assert result.used_tool is None
    assert result.answer == "Direct answer"


@pytest.mark.asyncio
async def test_judge_stage0_rejects():
    judge = Judge(JudgeConfig(stage0_enabled=True, stage1_client=None))
    result = await judge(
        query="question",
        candidate_answer="I don't know.",
        tool_evidence={},
        constraints={},
        trace_ctx={},
    )
    assert result.passed is False


@pytest.mark.asyncio
async def test_register_function_wraps_sync_callable():
    client = MockModelClient()
    client.queue_response(
        ModelResponse(
            text="TOOL=PINECONE",
            tokens=["TOOL=PINECONE"],
            token_logprobs=[-0.1],
            top_logprobs=[{"PINECONE": -0.1, "NONE": -2.0}],
            usage={"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
            raw={"mock": True},
        )
    )
    client.queue_text("Answer using Pinecone data")

    def pinecone_stub(query: str) -> dict:
        return {"matches": [{"id": "doc1", "score": 0.9}]}

    spec = ToolSpec(
        name="PINECONE",
        description="Vector search",
        input_schema={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        example_calls=[],
        cost_hint="medium",
        latency_hint_ms=10,
        side_effects=False,
        cacheable=False,
        cancellable=True,
    )

    tools = ToolRegistry()
    tools.register_function(pinecone_stub, spec)

    runtime = AgentRuntime(client=client, tools=tools, config=AgentConfig(allow_direct_answer_draft=False))
    result = await runtime.run("Search vector DB")

    assert result.used_tool == "PINECONE"


@pytest.mark.asyncio
async def test_multi_label_runs_multiple_tools():
    client = MockModelClient()
    client.queue_response(
        ModelResponse(
            text="TOOL=RAG",
            tokens=["TOOL=RAG"],
            token_logprobs=[-0.1],
            top_logprobs=[{"RAG": -0.1, "WEB": -0.2, "NONE": -2.0}],
            usage={"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
            raw={"mock": True},
        )
    )

    async def renderer(_query, tool_evidence, _metrics, _cost_tracker):
        assert isinstance(tool_evidence, dict)
        return "ok"

    tools = ToolRegistry()
    tools.register(DummyTool("RAG", delay_s=0.01))
    tools.register(DummyTool("WEB", delay_s=0.01))

    runtime = AgentRuntime(
        client=client,
        tools=tools,
        config=AgentConfig(
            allow_direct_answer_draft=False,
            multi_tool_mode="multi_label",
            answer_renderer=renderer,
        ),
    )

    result = await runtime.run("Benefits and reset my bamboohr password?")
    assert set(result.used_tools) == {"RAG", "WEB"}
    assert set(result.tool_results.keys()) == {"RAG", "WEB"}


@pytest.mark.asyncio
async def test_decompose_runs_multiple_tools():
    client = MockModelClient()
    client.queue_response(
        ModelResponse(
            text="SPLIT=YES",
            tokens=["SPLIT=YES"],
            token_logprobs=[-0.1],
            top_logprobs=[{"SPLIT=YES": -0.1, "SPLIT=NO": -1.5}],
            usage={"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
            raw={"mock": True},
        )
    )
    client.queue_text('["What is the policy on moonlighting?", "How do I reset my BambooHR password?"]')
    client.queue_response(
        ModelResponse(
            text="TOOL=RAG",
            tokens=["TOOL=RAG"],
            token_logprobs=[-0.1],
            top_logprobs=[{"RAG": -0.1, "WEB": -2.0, "NONE": -3.0}],
            usage={"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
            raw={"mock": True},
        )
    )
    client.queue_response(
        ModelResponse(
            text="TOOL=WEB",
            tokens=["TOOL=WEB"],
            token_logprobs=[-0.1],
            top_logprobs=[{"WEB": -0.1, "RAG": -2.0, "NONE": -3.0}],
            usage={"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
            raw={"mock": True},
        )
    )

    async def renderer(_query, tool_evidence, _metrics, _cost_tracker):
        assert isinstance(tool_evidence, dict)
        return "ok"

    tools = ToolRegistry()
    tools.register(DummyTool("RAG", delay_s=0.01))
    tools.register(DummyTool("WEB", delay_s=0.01))

    runtime = AgentRuntime(
        client=client,
        tools=tools,
        config=AgentConfig(
            allow_direct_answer_draft=False,
            multi_tool_mode="decompose",
            answer_renderer=renderer,
        ),
    )

    result = await runtime.run("Moonlighting policy and BambooHR password reset")
    assert set(result.used_tools) == {"RAG", "WEB"}


@pytest.mark.asyncio
async def test_tool_filter_removes_tool():
    client = MockModelClient()
    client.queue_response(
        ModelResponse(
            text="TOOL=RAG",
            tokens=["TOOL=RAG"],
            token_logprobs=[-0.1],
            top_logprobs=[{"RAG": -0.1, "WEB": -0.2, "NONE": -2.0}],
            usage={"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
            raw={"mock": True},
        )
    )
    client.queue_text("Answer using RAG evidence")

    tools = ToolRegistry()
    tools.register(DummyTool("RAG", delay_s=0.0))
    tools.register(DummyTool("WEB", delay_s=0.0))

    async def filter_tools(_query, specs, _client):
        return [spec for spec in specs if spec.name == "RAG"]

    runtime = AgentRuntime(
        client=client,
        tools=tools,
        config=AgentConfig(
            allow_direct_answer_draft=False,
            tool_filter=filter_tools,
        ),
    )

    result = await runtime.run("Need info")
    assert result.used_tool == "RAG"
