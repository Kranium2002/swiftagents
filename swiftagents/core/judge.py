from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .models import ModelClient
from .prompts import JUDGE_PROMPT


@dataclass
class JudgeResult:
    passed: bool
    score: float
    confidence: float
    reasons: List[str]


@dataclass
class JudgeConfig:
    stage0_enabled: bool = True
    stage1_client: Optional[ModelClient] = None
    stage2_client: Optional[ModelClient] = None
    stage1_threshold: float = 0.7
    stage2_threshold: float = 0.85
    timeout_s: float = 15.0
    requires_logprobs: bool = False
    max_tokens: int = 128
    temperature: float = 0.0
    prompt_template: Optional[str] = None


class Judge:
    def __init__(self, config: JudgeConfig) -> None:
        self._config = config

    async def __call__(
        self,
        *,
        query: str,
        candidate_answer: str,
        tool_evidence: Any,
        constraints: Dict[str, Any],
        trace_ctx: Dict[str, Any],
    ) -> JudgeResult:
        if self._config.stage0_enabled:
            stage0 = self._stage0_checks(candidate_answer)
            if stage0 is not None:
                return stage0

        if self._config.stage1_client is None:
            return JudgeResult(passed=True, score=1.0, confidence=0.5, reasons=["judge_disabled"])

        result = await self._run_llm_judge(
            client=self._config.stage1_client,
            query=query,
            candidate_answer=candidate_answer,
            tool_evidence=tool_evidence,
            constraints=constraints,
            trace_ctx=trace_ctx,
        )
        if result.passed and result.score >= self._config.stage2_threshold and self._config.stage2_client:
            result = await self._run_llm_judge(
                client=self._config.stage2_client,
                query=query,
                candidate_answer=candidate_answer,
                tool_evidence=tool_evidence,
                constraints=constraints,
                trace_ctx=trace_ctx,
            )
        return result

    def _stage0_checks(self, candidate_answer: str) -> Optional[JudgeResult]:
        if not candidate_answer.strip():
            return JudgeResult(passed=False, score=0.0, confidence=1.0, reasons=["empty_answer"])
        if "i don't know" in candidate_answer.lower():
            return JudgeResult(passed=False, score=0.2, confidence=0.6, reasons=["model_declined"])
        return None

    async def _run_llm_judge(
        self,
        *,
        client: ModelClient,
        query: str,
        candidate_answer: str,
        tool_evidence: Any,
        constraints: Dict[str, Any],
        trace_ctx: Dict[str, Any],
    ) -> JudgeResult:
        template = self._config.prompt_template or JUDGE_PROMPT
        prompt = template.format(
            query=query,
            candidate_answer=candidate_answer,
            tool_evidence=tool_evidence,
            constraints=constraints,
            trace_ctx=trace_ctx,
        )
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": "Return JSON only."},
        ]
        try:
            response = await asyncio.wait_for(
                client.complete(
                    messages,
                    max_tokens=self._config.max_tokens,
                    temperature=self._config.temperature,
                    logprobs=self._config.requires_logprobs,
                    top_logprobs=5,
                    response_format={"type": "json_object"},
                ),
                timeout=self._config.timeout_s,
            )
        except asyncio.TimeoutError:
            return JudgeResult(passed=False, score=0.0, confidence=0.2, reasons=["judge_timeout"])

        try:
            payload = json.loads(response.text)
        except json.JSONDecodeError:
            return JudgeResult(passed=False, score=0.0, confidence=0.3, reasons=["judge_invalid_json"])

        return JudgeResult(
            passed=bool(payload.get("pass", False)),
            score=float(payload.get("score", 0.0)),
            confidence=float(payload.get("confidence", 0.0)),
            reasons=list(payload.get("reasons", [])),
        )
