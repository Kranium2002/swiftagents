from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .models import ModelClient
from .prompts import TOOL_DECISION_PROMPT
from .tools import ToolSpec


@dataclass
class ToolDecision:
    label: str
    label_probs: Dict[str, float]
    entropy: float
    margin: float
    raw_text: str
    tokens: List[str]
    trace: Dict[str, Any]
    usage: Dict[str, int]


@dataclass
class RouterConfig:
    max_shortlist: int = 4
    top_logprobs: int = 20
    temperature: float = 0.0
    max_tokens: int = 6
    prompt_template: Optional[str] = None
    allow_none: bool = True


def shortlist_tools(query: str, tools: List[ToolSpec], max_tools: int = 4) -> List[ToolSpec]:
    if max_tools <= 0:
        return []
    if not tools:
        return []
    scores = []
    idf = _compute_idf([_tool_doc(t) for t in tools])
    q_vec = _tf_idf_vector(query, idf)
    for tool in tools:
        doc = _tool_doc(tool)
        t_vec = _tf_idf_vector(doc, idf)
        score = _cosine_sim(q_vec, t_vec)
        scores.append((score, tool))
    scores.sort(key=lambda x: x[0], reverse=True)
    return [tool for _, tool in scores[:max_tools]]


def build_tool_prompt(labels: List[str], prompt_template: Optional[str] = None) -> str:
    labels_str = "\n".join(f"- {label}" for label in labels)
    template = prompt_template or TOOL_DECISION_PROMPT
    return template.format(labels=labels_str)


class ToolRouter:
    def __init__(self, client: ModelClient, config: Optional[RouterConfig] = None) -> None:
        self._client = client
        self._config = config or RouterConfig()

    async def decide_tool(self, query: str, shortlist: List[ToolSpec]) -> ToolDecision:
        tool_labels = [tool.name for tool in shortlist]
        if not tool_labels:
            labels = ["NONE"]
        elif self._config.allow_none:
            labels = ["NONE"] + tool_labels
        else:
            labels = tool_labels
        prompt = build_tool_prompt(labels, self._config.prompt_template)
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": query},
        ]
        response = await self._client.complete(
            messages,
            max_tokens=self._config.max_tokens,
            temperature=self._config.temperature,
            logprobs=True,
            top_logprobs=self._config.top_logprobs,
        )
        raw_text = response.text.strip()
        label = _extract_label(raw_text)
        label = _normalize_label(label, labels)
        label_probs, entropy, margin, trace = _score_labels(
            labels=labels,
            raw_text=raw_text,
            tokens=response.tokens,
            token_logprobs=response.token_logprobs,
            top_logprobs=response.top_logprobs,
        )
        if label not in labels:
            label = labels[0] if labels else "NONE"
        return ToolDecision(
            label=label,
            label_probs=label_probs,
            entropy=entropy,
            margin=margin,
            raw_text=raw_text,
            tokens=response.tokens,
            trace=trace,
            usage=response.usage,
        )


def _extract_label(text: str) -> str:
    match = re.search(r"TOOL=([^\s]+)", text)
    if match:
        label = match.group(1).strip().strip("\"'").strip()
        return label.strip(".,;:")
    return "NONE"


def _normalize_label(label: str, labels: List[str]) -> str:
    if label in labels:
        return label
    normalized = label.strip().strip("\"'").strip().strip(".,;:").lower()
    label_map = {lab.lower(): lab for lab in labels}
    return label_map.get(normalized, label)


def _score_labels(
    *,
    labels: List[str],
    raw_text: str,
    tokens: List[str],
    token_logprobs: List[float],
    top_logprobs: Optional[List[Dict[str, float]]],
) -> Tuple[Dict[str, float], float, float, Dict[str, Any]]:
    trace: Dict[str, Any] = {"strategy": "top_logprobs_first_token"}
    label_logprobs: Dict[str, float] = {}
    label_start_index = _find_label_token_index(raw_text, tokens)
    if label_start_index is not None and top_logprobs:
        if label_start_index < len(top_logprobs):
            top = top_logprobs[label_start_index]
            for label in labels:
                if label in top:
                    label_logprobs[label] = top[label]
            trace["top_logprobs_used"] = True
        else:
            trace["top_logprobs_used"] = False
    else:
        trace["top_logprobs_used"] = False

    label = _extract_label(raw_text)
    if label in labels and label not in label_logprobs:
        label_logprobs[label] = _label_logprob_from_tokens(raw_text, tokens, token_logprobs)
        trace["used_output_logprob"] = True

    if not label_logprobs:
        fallback_label = labels[0] if labels else "NONE"
        label_logprobs = {fallback_label: 0.0}
        trace["fallback_uniform"] = True

    probs = _normalize_logprobs(label_logprobs)
    entropy = _entropy(list(probs.values()))
    margin = _margin(probs)
    return probs, entropy, margin, trace


def _label_logprob_from_tokens(raw_text: str, tokens: List[str], token_logprobs: List[float]) -> float:
    if not tokens or not token_logprobs:
        return 0.0
    joined = "".join(tokens)
    idx = joined.find(raw_text)
    if idx < 0:
        idx = 0
    # Approximate: sum logprobs for all tokens (best effort)
    return float(sum(token_logprobs))


def _find_label_token_index(raw_text: str, tokens: List[str]) -> Optional[int]:
    if not tokens:
        return None
    joined = "".join(tokens)
    label_idx = joined.find("TOOL=")
    if label_idx < 0:
        label_idx = 0
    label_start = label_idx + len("TOOL=")
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


def _entropy(probs: List[float]) -> float:
    ent = 0.0
    for p in probs:
        if p <= 0.0:
            continue
        ent -= p * math.log(p)
    return ent


def _margin(probs: Dict[str, float]) -> float:
    if not probs:
        return 0.0
    sorted_probs = sorted(probs.values(), reverse=True)
    if len(sorted_probs) == 1:
        return sorted_probs[0]
    return sorted_probs[0] - sorted_probs[1]


def _tool_doc(tool: ToolSpec) -> str:
    return f"{tool.name} {tool.description}"


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def _compute_idf(docs: List[str]) -> Dict[str, float]:
    df: Dict[str, int] = {}
    for doc in docs:
        seen = set(_tokenize(doc))
        for term in seen:
            df[term] = df.get(term, 0) + 1
    n_docs = max(len(docs), 1)
    idf: Dict[str, float] = {}
    for term, count in df.items():
        idf[term] = math.log(1 + n_docs / (1 + count))
    return idf


def _tf_idf_vector(text: str, idf: Dict[str, float]) -> Dict[str, float]:
    tf: Dict[str, int] = {}
    tokens = _tokenize(text)
    for tok in tokens:
        tf[tok] = tf.get(tok, 0) + 1
    vec: Dict[str, float] = {}
    for tok, count in tf.items():
        vec[tok] = (count / max(len(tokens), 1)) * idf.get(tok, 0.0)
    return vec


def _cosine_sim(v1: Dict[str, float], v2: Dict[str, float]) -> float:
    if not v1 or not v2:
        return 0.0
    dot = 0.0
    for tok, val in v1.items():
        dot += val * v2.get(tok, 0.0)
    norm1 = math.sqrt(sum(val * val for val in v1.values()))
    norm2 = math.sqrt(sum(val * val for val in v2.values()))
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return dot / (norm1 * norm2)
