from __future__ import annotations

import asyncio
import json
import math
import re
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from .cache import TTLCache, make_cache_key
from .judge import Judge, JudgeConfig, JudgeResult
from .metrics import CostTracker, Metrics, Trace
from .models import ModelClient
from .prompts import DECOMPOSE_PROMPT, FINAL_ANSWER_PROMPT, SPLIT_DECISION_PROMPT
from .router import RouterConfig, ToolDecision, ToolRouter, shortlist_tools
from .tools import ToolRegistry, ToolResult


@dataclass
class AgentConfig:
    max_parallel_tools: int = 1
    max_parallel_tools_uncertain: int = 2
    entropy_threshold: float = 0.8
    margin_threshold: float = 0.15
    tool_timeout_s: float = 15.0
    allow_speculative_side_effects: bool = False
    allow_direct_answer_draft: bool = True
    allow_direct_answer_commit: bool = False
    max_shortlist: int = 4
    answer_max_tokens: int = 512
    answer_temperature: float = 0.2
    answer_prompt_template: Optional[str] = None
    answer_renderer: Optional[
        Callable[[str, Optional[Any], Metrics, CostTracker], Awaitable[str]]
    ] = None
    decision_cache_ttl_s: float = 120.0
    tool_cache_ttl_s: float = 600.0
    cost_budget: Optional[float] = None
    model_cost_per_1k_tokens: float = 0.0
    tool_filter: Optional[
        Callable[[str, List[Any], ModelClient], Awaitable[List[Any]]]
    ] = None
    multi_tool_mode: str = "single"
    multi_intent_enabled: bool = False
    multi_intent_max_segments: int = 3
    multi_intent_min_segment_chars: int = 8
    multi_intent_allow_speculation: bool = False
    multi_intent_max_tools: int = 3
    multi_intent_merge_same_tool: bool = True
    multi_label_min_prob: float = 0.2
    multi_label_cum_prob: float = 0.8
    multi_label_max_tools: int = 3
    decompose_split_threshold: float = 0.6
    decompose_margin_threshold: float = 0.2
    decompose_max_subquestions: int = 3
    decompose_top_logprobs: int = 5
    decompose_decision_max_tokens: int = 6
    decompose_decision_temperature: float = 0.0
    decompose_prompt_template: Optional[str] = None
    decompose_decision_prompt_template: Optional[str] = None
    decompose_generation_max_tokens: int = 128
    decompose_generation_temperature: float = 0.2


@dataclass
class BranchResult:
    tool_name: Optional[str]
    tool_result: Optional[ToolResult]
    answer: Optional[str]
    error: Optional[str]
    latency_ms: float


@dataclass
class AgentResult:
    answer: str
    decision: ToolDecision
    used_tool: Optional[str]
    tool_results: Dict[str, ToolResult]
    judge_result: Optional[JudgeResult]
    metrics: Metrics
    trace: Trace
    used_tools: List[str] = field(default_factory=list)
    decisions: List[ToolDecision] = field(default_factory=list)


class AgentRuntime:
    def __init__(
        self,
        *,
        client: ModelClient,
        tools: ToolRegistry,
        config: Optional[AgentConfig] = None,
        router_config: Optional[RouterConfig] = None,
        judge: Optional[Judge] = None,
        decision_cache: Optional[TTLCache] = None,
        tool_cache: Optional[TTLCache] = None,
    ) -> None:
        self._client = client
        self._tools = tools
        self._config = config or AgentConfig()
        self._router = ToolRouter(client, router_config)
        self._judge = judge
        self._decision_cache = decision_cache or TTLCache(ttl_s=self._config.decision_cache_ttl_s)
        self._tool_cache = tool_cache or TTLCache(ttl_s=self._config.tool_cache_ttl_s)

    async def run(self, query: str) -> AgentResult:
        trace = Trace()
        metrics = Metrics()
        cost_tracker = CostTracker(budget=self._config.cost_budget)
        start = time.time()

        mode = self._config.multi_tool_mode
        if mode == "single" and self._config.multi_intent_enabled:
            mode = "multi_intent"

        if mode == "multi_intent":
            segments = _split_intents(
                query,
                max_segments=self._config.multi_intent_max_segments,
                min_chars=self._config.multi_intent_min_segment_chars,
            )
            if len(segments) > 1:
                (
                    answer,
                    used_tool,
                    used_tools,
                    tool_results,
                    judge_result,
                    decisions,
                ) = await self._run_multi_intent(query, segments, metrics, trace, cost_tracker)
                metrics.latency_ms = (time.time() - start) * 1000.0
                trace.add("final", data={"used_tool": used_tool, "used_tools": used_tools})
                return AgentResult(
                    answer=answer,
                    decision=decisions[0],
                    used_tool=used_tool,
                    tool_results=tool_results,
                    judge_result=judge_result,
                    metrics=metrics,
                    trace=trace,
                    used_tools=used_tools,
                    decisions=decisions,
                )
        elif mode == "multi_label":
            (
                answer,
                used_tool,
                used_tools,
                tool_results,
                judge_result,
                decisions,
            ) = await self._run_multi_label(query, metrics, trace, cost_tracker)
            metrics.latency_ms = (time.time() - start) * 1000.0
            trace.add("final", data={"used_tool": used_tool, "used_tools": used_tools})
            return AgentResult(
                answer=answer,
                decision=decisions[0],
                used_tool=used_tool,
                tool_results=tool_results,
                judge_result=judge_result,
                metrics=metrics,
                trace=trace,
                used_tools=used_tools,
                decisions=decisions,
            )
        elif mode == "decompose":
            (
                answer,
                used_tool,
                used_tools,
                tool_results,
                judge_result,
                decisions,
            ) = await self._run_decompose(query, metrics, trace, cost_tracker)
            metrics.latency_ms = (time.time() - start) * 1000.0
            trace.add("final", data={"used_tool": used_tool, "used_tools": used_tools})
            return AgentResult(
                answer=answer,
                decision=decisions[0],
                used_tool=used_tool,
                tool_results=tool_results,
                judge_result=judge_result,
                metrics=metrics,
                trace=trace,
                used_tools=used_tools,
                decisions=decisions,
            )

        tools = await self._apply_tool_filter(query, self._tools.list_specs(), trace)
        shortlist = shortlist_tools(query, tools, self._config.max_shortlist)
        trace.add("shortlist", data={"tools": [t.name for t in shortlist]})

        decision = await self._cached_decision(query, shortlist)
        metrics.model_calls += 1
        metrics.add_usage(decision.usage)
        trace.add(
            "decision",
            data={
                "label": decision.label,
                "entropy": decision.entropy,
                "margin": decision.margin,
                "probs": decision.label_probs,
            },
        )

        tool_candidates = self._select_tool_candidates(decision, shortlist, cost_tracker)
        uncertain = self._is_uncertain(decision)

        direct_task: Optional[asyncio.Task[str]] = None
        if uncertain and self._config.allow_direct_answer_draft:
            direct_task = asyncio.create_task(self._final_answer(query, None, metrics, cost_tracker))

        tool_tasks = []
        tool_results: Dict[str, ToolResult] = {}
        for name in tool_candidates:
            tool_tasks.append(asyncio.create_task(self._run_tool(name, query, metrics, trace, cost_tracker)))

        if tool_tasks:
            done, pending = await asyncio.wait(tool_tasks, timeout=self._config.tool_timeout_s)
            for task in pending:
                task.cancel()
            for task in done:
                result = task.result()
                if result.tool_name and result.tool_result:
                    tool_results[result.tool_name] = result.tool_result
        else:
            await asyncio.sleep(0)

        direct_answer: Optional[str] = None
        if direct_task:
            try:
                direct_answer = await direct_task
            except asyncio.CancelledError:
                direct_answer = None

        used_tool = None
        answer = ""
        judge_result = None
        if tool_results:
            used_tool, answer, judge_result = await self._commit_tool_answer(
                query,
                decision,
                tool_results,
                metrics,
                trace,
                direct_answer,
                cost_tracker,
            )
        else:
            answer = direct_answer or await self._final_answer(query, None, metrics, cost_tracker)

        metrics.latency_ms = (time.time() - start) * 1000.0
        trace.add("final", data={"used_tool": used_tool})
        return AgentResult(
            answer=answer,
            decision=decision,
            used_tool=used_tool,
            tool_results=tool_results,
            judge_result=judge_result,
            metrics=metrics,
            trace=trace,
            used_tools=[used_tool] if used_tool else [],
            decisions=[decision],
        )

    async def _cached_decision(self, query: str, shortlist: List[Any]) -> ToolDecision:
        cache_key = make_cache_key(
            {
                "query": query,
                "tools": ",".join(sorted(t.name for t in shortlist)),
            }
        )
        cached = self._decision_cache.get(cache_key)
        if cached is not None:
            return cached
        decision = await self._router.decide_tool(query, shortlist)
        self._decision_cache.set(cache_key, decision)
        return decision

    def _is_uncertain(self, decision: ToolDecision) -> bool:
        if decision.entropy > self._config.entropy_threshold:
            return True
        if decision.margin < self._config.margin_threshold:
            return True
        return False

    def _select_tool_candidates(
        self, decision: ToolDecision, shortlist: List[Any], cost_tracker: CostTracker
    ) -> List[str]:
        if not shortlist:
            return []
        if decision.label == "NONE" and not self._is_uncertain(decision):
            return []

        label_probs = list(decision.label_probs.items())
        label_probs.sort(key=lambda x: x[1], reverse=True)
        candidates = [label for label, _ in label_probs if label != "NONE"]

        if not self._is_uncertain(decision):
            return candidates[: self._config.max_parallel_tools]

        max_tools = min(self._config.max_parallel_tools_uncertain, len(shortlist))
        selected = []
        for label in candidates:
            if len(selected) >= max_tools:
                break
            tool = self._tools.get(label)
            if tool.spec.side_effects and not self._config.allow_speculative_side_effects:
                if label != decision.label:
                    continue
            if not _within_budget(tool.spec.cost_hint, cost_tracker):
                continue
            selected.append(label)
        return selected

    async def _run_multi_intent(
        self,
        query: str,
        segments: List[str],
        metrics: Metrics,
        trace: Trace,
        cost_tracker: CostTracker,
    ) -> Tuple[str, Optional[str], List[str], Dict[str, ToolResult], Optional[JudgeResult], List[ToolDecision]]:
        tools = self._tools.list_specs()
        tool_calls: List[Tuple[str, str]] = []
        decisions: List[ToolDecision] = []
        uncertain_any = False

        trace.add("multi_intent_segments", data={"segments": segments})

        for segment in segments:
            filtered_tools = await self._apply_tool_filter(segment, tools, trace)
            shortlist = shortlist_tools(segment, filtered_tools, self._config.max_shortlist)
            trace.add("shortlist", data={"segment": segment, "tools": [t.name for t in shortlist]})

            decision = await self._cached_decision(segment, shortlist)
            metrics.model_calls += 1
            metrics.add_usage(decision.usage)
            trace.add(
                "decision",
                data={
                    "segment": segment,
                    "label": decision.label,
                    "entropy": decision.entropy,
                    "margin": decision.margin,
                    "probs": decision.label_probs,
                },
            )
            decisions.append(decision)
            uncertain_any = uncertain_any or self._is_uncertain(decision)

            candidates = self._select_tools_for_decision(
                decision,
                shortlist,
                cost_tracker,
                allow_speculation=self._config.multi_intent_allow_speculation,
            )
            for name in candidates:
                tool_calls.append((name, segment))

        tool_plan = _merge_tool_calls(
            tool_calls,
            max_tools=self._config.multi_intent_max_tools,
            merge_same_tool=self._config.multi_intent_merge_same_tool,
        )
        trace.add(
            "tool_plan",
            data={"tools": [name for name, _ in tool_plan], "queries": {name: q for name, q in tool_plan}},
        )

        direct_task: Optional[asyncio.Task[str]] = None
        if uncertain_any and self._config.allow_direct_answer_draft:
            direct_task = asyncio.create_task(self._final_answer(query, None, metrics, cost_tracker))

        tool_results = await self._run_tool_plan(tool_plan, metrics, trace, cost_tracker, uncertain_any)

        direct_answer: Optional[str] = None
        if direct_task:
            try:
                direct_answer = await direct_task
            except asyncio.CancelledError:
                direct_answer = None

        used_tools, answer, judge_result = await self._commit_multi_tool_answer(
            query,
            tool_plan,
            tool_results,
            metrics,
            trace,
            direct_answer,
            cost_tracker,
        )
        used_tool = used_tools[0] if used_tools else None
        if not decisions:
            decisions = [ToolDecision(
                label="NONE",
                label_probs={"NONE": 1.0},
                entropy=0.0,
                margin=1.0,
                raw_text="TOOL=NONE",
                tokens=["TOOL=NONE"],
                trace={"strategy": "fallback"},
                usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            )]
        return answer, used_tool, used_tools, tool_results, judge_result, decisions

    async def _run_multi_label(
        self,
        query: str,
        metrics: Metrics,
        trace: Trace,
        cost_tracker: CostTracker,
    ) -> Tuple[str, Optional[str], List[str], Dict[str, ToolResult], Optional[JudgeResult], List[ToolDecision]]:
        tools = await self._apply_tool_filter(query, self._tools.list_specs(), trace)
        shortlist = shortlist_tools(query, tools, self._config.max_shortlist)
        trace.add("shortlist", data={"tools": [t.name for t in shortlist]})

        decision = await self._cached_decision(query, shortlist)
        metrics.model_calls += 1
        metrics.add_usage(decision.usage)
        trace.add(
            "decision",
            data={
                "label": decision.label,
                "entropy": decision.entropy,
                "margin": decision.margin,
                "probs": decision.label_probs,
            },
        )

        tool_candidates = self._select_multi_label_candidates(decision, shortlist, cost_tracker)
        trace.add("tool_plan", data={"tools": tool_candidates})
        uncertain = self._is_uncertain(decision)

        direct_task: Optional[asyncio.Task[str]] = None
        if uncertain and self._config.allow_direct_answer_draft:
            direct_task = asyncio.create_task(self._final_answer(query, None, metrics, cost_tracker))

        tool_plan = [(name, query) for name in tool_candidates]
        tool_results = await self._run_tool_plan(tool_plan, metrics, trace, cost_tracker, uncertain)

        direct_answer: Optional[str] = None
        if direct_task:
            try:
                direct_answer = await direct_task
            except asyncio.CancelledError:
                direct_answer = None

        used_tools, answer, judge_result = await self._commit_multi_tool_answer(
            query,
            tool_plan,
            tool_results,
            metrics,
            trace,
            direct_answer,
            cost_tracker,
        )
        used_tool = used_tools[0] if used_tools else None
        return answer, used_tool, used_tools, tool_results, judge_result, [decision]

    async def _run_decompose(
        self,
        query: str,
        metrics: Metrics,
        trace: Trace,
        cost_tracker: CostTracker,
    ) -> Tuple[str, Optional[str], List[str], Dict[str, ToolResult], Optional[JudgeResult], List[ToolDecision]]:
        should_split, split_trace, split_usage = await self._should_decompose(query)
        metrics.model_calls += 1
        metrics.add_usage(split_usage)
        trace.add("split_decision", data=split_trace)

        if not should_split:
            return await self._run_single(query, metrics, trace, cost_tracker)

        subquestions, gen_usage = await self._decompose_query(query)
        metrics.model_calls += 1
        metrics.add_usage(gen_usage)
        trace.add("decompose", data={"subquestions": subquestions})

        if len(subquestions) < 2:
            return await self._run_single(query, metrics, trace, cost_tracker)

        return await self._run_multi_intent(query, subquestions, metrics, trace, cost_tracker)

    async def _run_single(
        self,
        query: str,
        metrics: Metrics,
        trace: Trace,
        cost_tracker: CostTracker,
    ) -> Tuple[str, Optional[str], List[str], Dict[str, ToolResult], Optional[JudgeResult], List[ToolDecision]]:
        tools = await self._apply_tool_filter(query, self._tools.list_specs(), trace)
        shortlist = shortlist_tools(query, tools, self._config.max_shortlist)
        trace.add("shortlist", data={"tools": [t.name for t in shortlist]})

        decision = await self._cached_decision(query, shortlist)
        metrics.model_calls += 1
        metrics.add_usage(decision.usage)
        trace.add(
            "decision",
            data={
                "label": decision.label,
                "entropy": decision.entropy,
                "margin": decision.margin,
                "probs": decision.label_probs,
            },
        )

        tool_candidates = self._select_tool_candidates(decision, shortlist, cost_tracker)
        uncertain = self._is_uncertain(decision)

        direct_task: Optional[asyncio.Task[str]] = None
        if uncertain and self._config.allow_direct_answer_draft:
            direct_task = asyncio.create_task(self._final_answer(query, None, metrics, cost_tracker))

        tool_tasks = []
        tool_results: Dict[str, ToolResult] = {}
        for name in tool_candidates:
            tool_tasks.append(asyncio.create_task(self._run_tool(name, query, metrics, trace, cost_tracker)))

        if tool_tasks:
            done, pending = await asyncio.wait(tool_tasks, timeout=self._config.tool_timeout_s)
            for task in pending:
                task.cancel()
            for task in done:
                result = task.result()
                if result.tool_name and result.tool_result:
                    tool_results[result.tool_name] = result.tool_result
        else:
            await asyncio.sleep(0)

        direct_answer: Optional[str] = None
        if direct_task:
            try:
                direct_answer = await direct_task
            except asyncio.CancelledError:
                direct_answer = None

        used_tool = None
        answer = ""
        judge_result = None
        if tool_results:
            used_tool, answer, judge_result = await self._commit_tool_answer(
                query,
                decision,
                tool_results,
                metrics,
                trace,
                direct_answer,
                cost_tracker,
            )
        else:
            answer = direct_answer or await self._final_answer(query, None, metrics, cost_tracker)

        used_tools = [used_tool] if used_tool else []
        return answer, used_tool, used_tools, tool_results, judge_result, [decision]

    async def _should_decompose(
        self,
        query: str,
    ) -> Tuple[bool, Dict[str, Any], Dict[str, int]]:
        prompt_template = self._config.decompose_decision_prompt_template or SPLIT_DECISION_PROMPT
        messages = [
            {"role": "system", "content": prompt_template},
            {"role": "user", "content": query},
        ]
        response = await self._client.complete(
            messages,
            max_tokens=self._config.decompose_decision_max_tokens,
            temperature=self._config.decompose_decision_temperature,
            logprobs=True,
            top_logprobs=self._config.decompose_top_logprobs,
        )
        raw_text = response.text.strip()
        label = _extract_split_label(raw_text)
        labels = ["YES", "NO"]
        label_logprobs: Dict[str, float] = {}
        idx = _find_prefix_token_index("SPLIT=", response.tokens)
        if idx is not None and response.top_logprobs:
            if idx < len(response.top_logprobs):
                top = response.top_logprobs[idx]
                for option in labels:
                    if option in top:
                        label_logprobs[option] = top[option]
                    pref = f"SPLIT={option}"
                    if pref in top:
                        label_logprobs[option] = top[pref]
        if not label_logprobs and label in labels:
            label_logprobs[label] = float(sum(response.token_logprobs)) if response.token_logprobs else 0.0
        if not label_logprobs:
            label_logprobs = {"NO": 0.0}

        probs = _normalize_logprobs(label_logprobs)
        entropy = _entropy_from_probs(list(probs.values()))
        margin = _margin_from_probs(probs)
        p_yes = probs.get("YES", 0.0)
        should_split = (
            p_yes >= self._config.decompose_split_threshold
            and margin >= self._config.decompose_margin_threshold
        )
        trace = {
            "label": label,
            "probs": probs,
            "entropy": entropy,
            "margin": margin,
            "raw_text": raw_text,
        }
        return should_split, trace, response.usage

    async def _decompose_query(
        self,
        query: str,
    ) -> Tuple[List[str], Dict[str, int]]:
        prompt_template = self._config.decompose_prompt_template or DECOMPOSE_PROMPT
        prompt = prompt_template.format(max_subquestions=self._config.decompose_max_subquestions)
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": query},
        ]
        response = await self._client.complete(
            messages,
            max_tokens=self._config.decompose_generation_max_tokens,
            temperature=self._config.decompose_generation_temperature,
            logprobs=False,
            top_logprobs=0,
        )
        subquestions = _parse_subquestions(response.text)
        if self._config.decompose_max_subquestions > 0:
            subquestions = subquestions[: self._config.decompose_max_subquestions]
        return subquestions, response.usage

    def _select_multi_label_candidates(
        self,
        decision: ToolDecision,
        shortlist: List[Any],
        cost_tracker: CostTracker,
    ) -> List[str]:
        if not shortlist:
            return []
        if decision.label == "NONE" and not self._is_uncertain(decision):
            return []

        label_probs = list(decision.label_probs.items())
        label_probs = [(label, prob) for label, prob in label_probs if label != "NONE"]
        label_probs.sort(key=lambda x: x[1], reverse=True)
        if not label_probs:
            return []

        max_tools = min(self._config.multi_label_max_tools, len(shortlist))
        min_prob = self._config.multi_label_min_prob
        cum_prob_target = self._config.multi_label_cum_prob

        selected: List[str] = []
        cumulative = 0.0
        for label, prob in label_probs:
            if len(selected) >= max_tools:
                break
            if prob < min_prob and cumulative >= cum_prob_target and selected:
                break

            tool = self._tools.get(label)
            if tool.spec.side_effects and not self._config.allow_speculative_side_effects:
                if label != decision.label:
                    continue
            if not _within_budget(tool.spec.cost_hint, cost_tracker):
                continue

            selected.append(label)
            cumulative += prob
        return selected

    def _select_tools_for_decision(
        self,
        decision: ToolDecision,
        shortlist: List[Any],
        cost_tracker: CostTracker,
        *,
        allow_speculation: bool,
    ) -> List[str]:
        if not shortlist:
            return []
        if decision.label == "NONE" and not self._is_uncertain(decision):
            return []

        label_probs = list(decision.label_probs.items())
        label_probs.sort(key=lambda x: x[1], reverse=True)
        candidates = [label for label, _ in label_probs if label != "NONE"]
        if not candidates:
            return []

        if not self._is_uncertain(decision) or not allow_speculation:
            primary = decision.label if decision.label != "NONE" else candidates[0]
            if primary == "NONE":
                return []
            tool = self._tools.get(primary)
            if not _within_budget(tool.spec.cost_hint, cost_tracker):
                return []
            return [primary]

        max_tools = min(self._config.max_parallel_tools_uncertain, len(shortlist))
        selected = []
        for label in candidates:
            if len(selected) >= max_tools:
                break
            tool = self._tools.get(label)
            if tool.spec.side_effects and not self._config.allow_speculative_side_effects:
                if label != decision.label:
                    continue
            if not _within_budget(tool.spec.cost_hint, cost_tracker):
                continue
            selected.append(label)
        return selected

    async def _run_tool_plan(
        self,
        tool_plan: List[Tuple[str, str]],
        metrics: Metrics,
        trace: Trace,
        cost_tracker: CostTracker,
        uncertain_any: bool,
    ) -> Dict[str, ToolResult]:
        if not tool_plan:
            return {}

        max_parallel = self._config.max_parallel_tools_uncertain if uncertain_any else self._config.max_parallel_tools
        max_parallel = max(1, max_parallel)
        semaphore = asyncio.Semaphore(max_parallel)

        async def _run_with_limit(name: str, q: str) -> BranchResult:
            async with semaphore:
                return await self._run_tool(name, q, metrics, trace, cost_tracker)

        tasks = [asyncio.create_task(_run_with_limit(name, q)) for name, q in tool_plan]
        results = await asyncio.gather(*tasks)

        tool_results: Dict[str, ToolResult] = {}
        for result in results:
            if result.tool_name and result.tool_result:
                tool_results[result.tool_name] = result.tool_result
        return tool_results

    async def _apply_tool_filter(
        self,
        query: str,
        tools: List[Any],
        trace: Trace,
    ) -> List[Any]:
        tool_filter = self._config.tool_filter
        if tool_filter is None:
            return tools
        result = tool_filter(query, tools, self._client)
        if asyncio.iscoroutine(result):
            filtered = await result
        else:
            filtered = result
        trace.add(
            "tool_filter",
            data={"query": query, "tools": [tool.name for tool in filtered]},
        )
        return filtered

    async def _commit_multi_tool_answer(
        self,
        query: str,
        tool_plan: List[Tuple[str, str]],
        tool_results: Dict[str, ToolResult],
        metrics: Metrics,
        trace: Trace,
        direct_answer: Optional[str],
        cost_tracker: CostTracker,
    ) -> Tuple[List[str], str, Optional[JudgeResult]]:
        total_calls = len(tool_plan)
        if not tool_results:
            if total_calls:
                metrics.tool_calls_wasted += total_calls
            answer = direct_answer or await self._final_answer(query, None, metrics, cost_tracker)
            return [], answer, None

        ok_results = {name: result for name, result in tool_results.items() if result.ok}
        if not ok_results:
            metrics.tool_calls_wasted += total_calls
            if direct_answer:
                return [], direct_answer, None
            return [], "Unable to produce an answer with available tools.", None

        combined_evidence = {name: result.data for name, result in ok_results.items()}
        answer = await self._final_answer(query, combined_evidence, metrics, cost_tracker)

        judge = self._judge
        requires_judge = any(result.metadata.get("requires_judge") for result in ok_results.values())
        if judge is None and requires_judge:
            judge = Judge(JudgeConfig(stage0_enabled=True, stage1_client=None))

        judge_result = None
        if judge is not None:
            judge_result = await judge(
                query=query,
                candidate_answer=answer,
                tool_evidence=combined_evidence,
                constraints={},
                trace_ctx={"multi_intent": True},
            )
            trace.add(
                "judge_result",
                data={"tool": "MULTI", "passed": judge_result.passed, "score": judge_result.score},
            )
            if not judge_result.passed and direct_answer and self._config.allow_direct_answer_commit:
                metrics.tool_calls_wasted += total_calls
                return [], direct_answer, judge_result

        used_tools = [name for name, _ in tool_plan if name in ok_results]
        metrics.tool_calls_used += len(used_tools)
        metrics.tool_calls_wasted += max(total_calls - len(used_tools), 0)
        return used_tools, answer, judge_result

    async def _run_tool(
        self,
        name: str,
        query: str,
        metrics: Metrics,
        trace: Trace,
        cost_tracker: CostTracker,
    ) -> BranchResult:
        tool = self._tools.get(name)
        cache_key = make_cache_key({"tool": name, "query": query})
        if tool.spec.cacheable:
            cached = self._tool_cache.get(cache_key)
            if cached is not None:
                metrics.tool_calls += 1
                trace.add("tool_cache_hit", data={"tool": name})
                return BranchResult(tool_name=name, tool_result=cached, answer=None, error=None, latency_ms=0.0)

        start = time.time()
        metrics.tool_calls += 1
        try:
            self._tools.validate_args(tool.spec, {"query": query})
            result = await asyncio.wait_for(tool.handler(query=query), timeout=self._config.tool_timeout_s)
        except asyncio.TimeoutError:
            trace.add("tool_timeout", data={"tool": name})
            return BranchResult(tool_name=name, tool_result=None, answer=None, error="timeout", latency_ms=0.0)
        except Exception as exc:  # pragma: no cover
            trace.add("tool_error", data={"tool": name, "error": str(exc)})
            return BranchResult(tool_name=name, tool_result=None, answer=None, error=str(exc), latency_ms=0.0)

        latency_ms = (time.time() - start) * 1000.0
        if result.metadata is not None and "query" not in result.metadata:
            result.metadata["query"] = query
        if tool.spec.cacheable and result.ok:
            self._tool_cache.set(cache_key, result)
        cost_tracker.add_tool_cost(name, _tool_cost(tool.spec.cost_hint))
        trace.add("tool_complete", data={"tool": name, "ok": result.ok}, duration_ms=latency_ms)
        return BranchResult(tool_name=name, tool_result=result, answer=None, error=None, latency_ms=latency_ms)

    async def _final_answer(
        self,
        query: str,
        tool_evidence: Optional[Any],
        metrics: Metrics,
        cost_tracker: CostTracker,
    ) -> str:
        if self._config.answer_renderer is not None:
            return await self._config.answer_renderer(query, tool_evidence, metrics, cost_tracker)
        evidence_str = ""
        if tool_evidence is not None:
            evidence_str = json.dumps(tool_evidence, ensure_ascii=True, indent=2)
        prompt_template = self._config.answer_prompt_template or FINAL_ANSWER_PROMPT
        prompt = prompt_template.format(query=query, tool_evidence=evidence_str)
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": query},
        ]
        response = await self._client.complete(
            messages,
            max_tokens=self._config.answer_max_tokens,
            temperature=self._config.answer_temperature,
            logprobs=False,
            top_logprobs=0,
        )
        metrics.model_calls += 1
        metrics.add_usage(response.usage)
        if self._config.model_cost_per_1k_tokens > 0.0:
            total_tokens = response.usage.get("total_tokens", 0)
            cost = (total_tokens / 1000.0) * self._config.model_cost_per_1k_tokens
            cost_tracker.add_model_cost(cost)
        return response.text.strip()

    async def _commit_tool_answer(
        self,
        query: str,
        decision: ToolDecision,
        tool_results: Dict[str, ToolResult],
        metrics: Metrics,
        trace: Trace,
        direct_answer: Optional[str],
        cost_tracker: CostTracker,
    ) -> Tuple[Optional[str], str, Optional[JudgeResult]]:
        candidates = []
        sorted_tools = sorted(
            tool_results.items(),
            key=lambda item: decision.label_probs.get(item[0], 0.0),
            reverse=True,
        )
        for tool_name, result in sorted_tools:
            if not result.ok:
                continue
            answer = await self._final_answer(query, result.data, metrics, cost_tracker)
            candidates.append((tool_name, answer, result))

        if not candidates and direct_answer:
            return None, direct_answer, None
        if not candidates:
            metrics.tool_calls_wasted += len(tool_results)
            return None, "Unable to produce an answer with available tools.", None

        judge = self._judge
        requires_judge = any(res.metadata.get("requires_judge") for _, _, res in candidates)
        if judge is None and requires_judge:
            judge = Judge(JudgeConfig(stage0_enabled=True, stage1_client=None))

        if judge is None:
            tool_name, answer, _ = candidates[0]
            metrics.tool_calls_used += 1
            metrics.tool_calls_wasted += max(len(tool_results) - 1, 0)
            return tool_name, answer, None

        best: Tuple[Optional[str], str, Optional[JudgeResult]] = (None, "", None)
        for tool_name, answer, result in candidates:
            judge_result = await judge(
                query=query,
                candidate_answer=answer,
                tool_evidence=result.data,
                constraints={},
                trace_ctx={"decision": decision.trace},
            )
            trace.add(
                "judge_result",
                data={"tool": tool_name, "passed": judge_result.passed, "score": judge_result.score},
            )
            if judge_result.passed:
                metrics.tool_calls_used += 1
                metrics.tool_calls_wasted += max(len(tool_results) - 1, 0)
                return tool_name, answer, judge_result
            if best[2] is None or judge_result.score > best[2].score:
                best = (tool_name, answer, judge_result)

        if best[0] is not None:
            metrics.tool_calls_used += 1
            metrics.tool_calls_wasted += max(len(tool_results) - 1, 0)
            return best

        if direct_answer and self._config.allow_direct_answer_commit:
            return None, direct_answer, None

        tool_name, answer, _ = candidates[0]
        metrics.tool_calls_used += 1
        metrics.tool_calls_wasted += max(len(tool_results) - 1, 0)
        return tool_name, answer, None


def _within_budget(cost_hint: str, cost_tracker: CostTracker) -> bool:
    if cost_tracker.budget is None:
        return True
    cost = _tool_cost(cost_hint)
    return (cost_tracker.total_cost + cost) <= cost_tracker.budget


def _tool_cost(cost_hint: str) -> float:
    cost_map = {"cheap": 0.1, "medium": 0.5, "expensive": 1.0}
    return cost_map.get(cost_hint, 0.5)


def _merge_tool_calls(
    tool_calls: List[Tuple[str, str]],
    *,
    max_tools: int,
    merge_same_tool: bool,
) -> List[Tuple[str, str]]:
    if not tool_calls:
        return []

    if merge_same_tool:
        merged: Dict[str, List[str]] = {}
        order: List[str] = []
        for name, query in tool_calls:
            if name not in merged:
                merged[name] = [query]
                order.append(name)
            else:
                merged[name].append(query)
        plan = [(name, "; ".join(merged[name])) for name in order]
    else:
        plan = list(tool_calls)

    if max_tools > 0:
        plan = plan[:max_tools]
    return plan


def _split_intents(query: str, *, max_segments: int, min_chars: int) -> List[str]:
    if not query:
        return []

    parts = [chunk.strip() for chunk in re.split(r"[?;\n]+", query) if chunk.strip()]
    connectors = [" and ", " & ", " plus ", " also ", " then "]
    expanded: List[str] = []
    for part in parts:
        sub_parts = [part]
        for token in connectors:
            next_parts: List[str] = []
            for segment in sub_parts:
                if token in segment:
                    next_parts.extend([seg.strip() for seg in segment.split(token) if seg.strip()])
                else:
                    next_parts.append(segment)
            sub_parts = next_parts
        expanded.extend(sub_parts)

    segments: List[str] = []
    for segment in expanded:
        if _should_keep_segment(segment, min_chars):
            segments.append(segment)

    if not segments:
        return [query.strip()]

    seen = set()
    ordered: List[str] = []
    for segment in segments:
        if segment in seen:
            continue
        seen.add(segment)
        ordered.append(segment)
        if max_segments > 0 and len(ordered) >= max_segments:
            break
    return ordered


def _should_keep_segment(segment: str, min_chars: int) -> bool:
    if len(segment) >= min_chars:
        return True
    tokens = re.findall(r"[A-Za-z0-9]+", segment)
    return any(len(token) >= 3 for token in tokens)


def _extract_split_label(text: str) -> str:
    match = re.search(r"SPLIT=([^\s]+)", text)
    if match:
        label = match.group(1).strip().strip("\"'").strip()
        return label.strip(".,;:").upper()
    return "NO"


def _find_prefix_token_index(prefix: str, tokens: List[str]) -> Optional[int]:
    if not tokens:
        return None
    joined = "".join(tokens)
    prefix_idx = joined.find(prefix)
    if prefix_idx < 0:
        prefix_idx = 0
    label_start = prefix_idx + len(prefix)
    offset = 0
    for i, token in enumerate(tokens):
        next_offset = offset + len(token)
        if offset <= label_start < next_offset:
            return i
        offset = next_offset
    return None


def _normalize_logprobs(label_logprobs: Dict[str, float]) -> Dict[str, float]:
    max_lp = max(label_logprobs.values())
    exp_vals = {k: math.exp(v - max_lp) for k, v in label_logprobs.items()}
    total = sum(exp_vals.values())
    if total == 0.0:
        return {k: 1.0 / len(exp_vals) for k in exp_vals}
    return {k: v / total for k, v in exp_vals.items()}


def _entropy_from_probs(probs: List[float]) -> float:
    ent = 0.0
    for p in probs:
        if p <= 0.0:
            continue
        ent -= p * math.log(p)
    return ent


def _margin_from_probs(probs: Dict[str, float]) -> float:
    if not probs:
        return 0.0
    sorted_probs = sorted(probs.values(), reverse=True)
    if len(sorted_probs) == 1:
        return sorted_probs[0]
    return sorted_probs[0] - sorted_probs[1]


def _parse_subquestions(text: str) -> List[str]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return []
    if isinstance(data, list):
        items = [str(item).strip() for item in data if str(item).strip()]
    else:
        return []
    seen = set()
    result: List[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result
