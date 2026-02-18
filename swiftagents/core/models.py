from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol

import httpx


class BackendDoesNotSupportLogprobsError(RuntimeError):
    pass


class ModelClientError(RuntimeError):
    pass


@dataclass
class ModelResponse:
    text: str
    tokens: List[str]
    token_logprobs: List[float]
    top_logprobs: Optional[List[Dict[str, float]]]
    usage: Dict[str, int]
    raw: Any


class ModelClient(Protocol):
    async def complete(
        self,
        messages: List[Dict[str, str]],
        *,
        max_tokens: int,
        temperature: float,
        logprobs: bool,
        top_logprobs: int,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> ModelResponse:
        ...


class _OpenAIBaseClient:
    def __init__(
        self,
        *,
        api_key: Optional[str],
        base_url: str,
        model: str,
        timeout_s: float = 30.0,
        headers: Optional[Dict[str, str]] = None,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout_s = timeout_s
        self._headers = headers or {}
        self._client = httpx.AsyncClient(timeout=timeout_s)

    async def close(self) -> None:
        await self._client.aclose()

    async def complete(
        self,
        messages: List[Dict[str, str]],
        *,
        max_tokens: int,
        temperature: float,
        logprobs: bool,
        top_logprobs: int,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> ModelResponse:
        payload: Dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "max_completion_tokens": max_tokens,
            "temperature": temperature,
        }
        if response_format is not None:
            payload["response_format"] = response_format
        if logprobs:
            payload["logprobs"] = True
            payload["top_logprobs"] = top_logprobs

        headers = dict(self._headers)
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        url = f"{self._base_url}/chat/completions"
        try:
            resp = await self._client.post(url, json=payload, headers=headers)
        except httpx.HTTPError as exc:
            raise ModelClientError(str(exc)) from exc

        if resp.status_code == 400 and "max_completion_tokens" in resp.text:
            payload.pop("max_completion_tokens")
            payload["max_tokens"] = max_tokens
            try:
                resp = await self._client.post(url, json=payload, headers=headers)
            except httpx.HTTPError as exc:
                raise ModelClientError(str(exc)) from exc

        if resp.status_code >= 400:
            raise ModelClientError(f"HTTP {resp.status_code}: {resp.text}")

        data = resp.json()
        return _parse_openai_response(data, require_logprobs=logprobs)


class OpenAIChatCompletionsClient(_OpenAIBaseClient):
    def __init__(
        self,
        *,
        api_key: Optional[str],
        model: str,
        base_url: str = "https://api.openai.com/v1",
        timeout_s: float = 30.0,
        headers: Optional[Dict[str, str]] = None,
    ) -> None:
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            model=model,
            timeout_s=timeout_s,
            headers=headers,
        )


class VLLMOpenAICompatibleClient(_OpenAIBaseClient):
    def __init__(
        self,
        *,
        api_key: Optional[str],
        model: str,
        base_url: str = "http://localhost:8000/v1",
        timeout_s: float = 30.0,
        headers: Optional[Dict[str, str]] = None,
    ) -> None:
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            model=model,
            timeout_s=timeout_s,
            headers=headers,
        )


class MockModelClient:
    def __init__(self) -> None:
        self._queue: List[ModelResponse] = []
        self.calls: List[Dict[str, Any]] = []

    def queue_response(self, response: ModelResponse) -> None:
        self._queue.append(response)

    def queue_text(
        self,
        text: str,
        *,
        token_logprobs: Optional[List[float]] = None,
        top_logprobs: Optional[List[Dict[str, float]]] = None,
        usage: Optional[Dict[str, int]] = None,
    ) -> None:
        tokens = _simple_tokenize(text)
        if token_logprobs is None:
            token_logprobs = [0.0 for _ in tokens]
        if usage is None:
            usage = {
                "prompt_tokens": 0,
                "completion_tokens": len(tokens),
                "total_tokens": len(tokens),
            }
        response = ModelResponse(
            text=text,
            tokens=tokens,
            token_logprobs=token_logprobs,
            top_logprobs=top_logprobs,
            usage=usage,
            raw={"mock": True},
        )
        self.queue_response(response)

    async def complete(
        self,
        messages: List[Dict[str, str]],
        *,
        max_tokens: int,
        temperature: float,
        logprobs: bool,
        top_logprobs: int,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> ModelResponse:
        await asyncio.sleep(0)
        self.calls.append(
            {
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "logprobs": logprobs,
                "top_logprobs": top_logprobs,
                "response_format": response_format,
            }
        )
        if not self._queue:
            raise ModelClientError("MockModelClient queue empty")
        response = self._queue.pop(0)
        if logprobs and not response.token_logprobs:
            raise BackendDoesNotSupportLogprobsError("Mock response missing logprobs")
        return response


def _parse_openai_response(data: Dict[str, Any], *, require_logprobs: bool) -> ModelResponse:
    choices = data.get("choices") or []
    if not choices:
        raise ModelClientError("Empty response choices")
    choice = choices[0]
    message = choice.get("message") or {}
    text = message.get("content") or ""
    logprobs = choice.get("logprobs")
    if require_logprobs and not logprobs:
        raise BackendDoesNotSupportLogprobsError("Backend did not return logprobs")

    tokens: List[str] = []
    token_logprobs: List[float] = []
    top_logprobs: Optional[List[Dict[str, float]]] = None
    if logprobs and isinstance(logprobs, dict):
        content = logprobs.get("content") or []
        top_logprobs = []
        for entry in content:
            tokens.append(entry.get("token", ""))
            token_logprobs.append(float(entry.get("logprob", 0.0)))
            tps = entry.get("top_logprobs") or []
            tps_map: Dict[str, float] = {}
            for alt in tps:
                token = alt.get("token")
                lp = alt.get("logprob")
                if token is None or lp is None:
                    continue
                tps_map[token] = float(lp)
            top_logprobs.append(tps_map)

    usage = data.get("usage") or {}
    return ModelResponse(
        text=text,
        tokens=tokens,
        token_logprobs=token_logprobs,
        top_logprobs=top_logprobs,
        usage={
            "prompt_tokens": int(usage.get("prompt_tokens", 0)),
            "completion_tokens": int(usage.get("completion_tokens", 0)),
            "total_tokens": int(usage.get("total_tokens", 0)),
        },
        raw=data,
    )


def _simple_tokenize(text: str) -> List[str]:
    if not text:
        return []
    parts: List[str] = []
    buf = []
    for ch in text:
        if ch.isspace():
            if buf:
                parts.append("".join(buf))
                buf = []
            parts.append(ch)
        else:
            buf.append(ch)
    if buf:
        parts.append("".join(buf))
    return parts
