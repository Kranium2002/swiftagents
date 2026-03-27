# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

**swiftagents** — a logprob-native, async-first agent runtime for LLM tool routing and execution. Python ≥ 3.10, Apache-2.0.

## Commands

```bash
pip install -e .[dev]                                    # install for local dev
pytest                                                   # run all tests
pytest swiftagents/tests/test_router.py                  # run a single test file
pytest swiftagents/tests/test_router.py::test_shortlist_prefers_relevant_tools  # single test
python -m swiftagents.examples.tool_selection             # run example
python -m swiftagents.benchmarks.run_benchmark            # run benchmarks
```

pytest is configured with `asyncio_mode = "auto"` in pyproject.toml — async test functions work without the `@pytest.mark.asyncio` decorator.

## Architecture

All public API is exported from `swiftagents.core` (`swiftagents/core/__init__.py`).

### Execution flow

`AgentRuntime.run(query)` in `core/scheduler.py` orchestrates the full pipeline:

1. **Shortlist** — `ToolRouter` (`core/router.py`) uses TF-IDF similarity to narrow the tool registry to ≤4 candidates.
2. **Route** — The same LLM is prompted to output `TOOL=<LABEL>`. Logprobs on the first token score all candidate labels. Returns a `ToolDecision` with entropy and margin for uncertainty quantification.
3. **Speculate** — If uncertain (entropy > 0.8 or margin < 0.15), the runtime speculatively executes up to 2 tools in parallel, bounded to prevent runaway cost. Side-effecting tools are never speculated unless explicitly allowed.
4. **Execute** — Tools run async with per-tool timeout (15s default) and semaphore-bounded concurrency.
5. **Answer** — LLM synthesizes a final answer from tool evidence.
6. **Judge** (optional) — `Judge` (`core/judge.py`) validates the answer in a 3-stage pipeline: deterministic checks → cheap LLM → optional stronger LLM escalation.

### Multi-tool routing modes (`AgentConfig.multi_tool_mode`)

- `single` (default): one route decision, bounded speculation when uncertain.
- `multi_label`: one router call, multiple tools selected via logprob thresholds.
- `multi_intent`: heuristic query splitting into segments, route each independently.
- `decompose`: LLM-driven decomposition into sub-questions, then route each.

### Key modules

| Module | Role |
|---|---|
| `core/scheduler.py` | `AgentRuntime`, `AgentConfig`, `AgentResult` — main orchestrator (~1100 lines) |
| `core/router.py` | `ToolRouter` — TF-IDF shortlisting + logprob-based label scoring |
| `core/models.py` | `ModelClient` protocol + `OpenAIChatCompletionsClient`, `VLLMOpenAICompatibleClient`, `MockModelClient` |
| `core/tools.py` | `ToolSpec`, `ToolRegistry`, `ToolResult` — tool registration (class-based or function-based) |
| `core/judge.py` | `Judge` — 3-stage answer validation pipeline |
| `core/cache.py` | `TTLCache` — OrderedDict-based LRU cache with per-entry TTL |
| `core/metrics.py` | `Metrics` + `Trace` — token usage, latency, wasted-work tracking |
| `core/prompts.py` | Prompt templates for routing, decomposition, answering, judging |

### Design conventions

- **Logprobs are mandatory.** ModelClient backends must return token-level logprobs; the system hard-errors otherwise (`BackendDoesNotSupportLogprobsError`).
- **Tool labels should be short uppercase names** (e.g. `WEB`, `RAG`, `PINECONE`) for stable routing extraction via `TOOL=<LABEL>` regex.
- **Protocol-based interfaces** — `ModelClient` uses `typing.Protocol`, not ABC.
- **Dataclasses for configuration** — `AgentConfig`, `RouterConfig`, `JudgeConfig`, `ToolSpec` are all dataclasses.
- **Sync functions auto-wrapped** — `ToolRegistry.register_function()` wraps sync callables with `asyncio.to_thread()`.
- **MockModelClient** — queue responses with `queue_text()` / `queue_response()` for deterministic testing.
