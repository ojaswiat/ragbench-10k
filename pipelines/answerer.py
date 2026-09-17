"""Turns a query plus its retrieved nodes into a cited answer.

All three pipelines share one Answerer so that score differences come from
retrieval rather than generation. The prompt gets the query text and the
nodes' ids and content, nothing else.
"""
import re
import time
from dataclasses import dataclass

from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.schema import NodeWithScore

import llm_client.config as config
from llm_client.llm_factory import LLMFactory

_STAGE = "answerer"

# Markers are kept in pipeline_output rather than stripped, so a stored
# answer stays auditable against its own citations.
#
# Fullwidth brackets are accepted alongside ASCII ones. Some models emit
# U+3010/U+3011 for a doubled square bracket regardless of what the system
# prompt asks for, and a citation lost to a delimiter substitution would
# be scored as an uncited claim by the Citation Audit.
_CITATION_PATTERN = re.compile(r"(?:\[\[|\u3010)node:([\w\-]+)(?:\]\]|\u3011)")

_SYSTEM_PROMPT = (
    "You answer questions using only the numbered sources provided. "
    "Do not use outside knowledge, and do not guess: if the sources do not "
    "contain the answer, say so plainly. "
    "After any claim drawn from a source, immediately append "
    "[[node:<node_id>]] citing the exact node it came from. "
    "Write that marker with plain ASCII square brackets, U+005B and "
    "U+005D, doubled on each side. Never substitute fullwidth or CJK "
    "bracket characters. "
    "Use the node_id exactly as given. Answer concisely."
)


@dataclass
class AnswerResult:
    raw_text: str
    cited_node_ids: list[str]
    input_tokens: int
    output_tokens: int
    latency_sec: float


def parse_citations(raw_text: str) -> list[str]:
    """Node ids cited in `raw_text`, deduplicated, first-appearance order."""
    seen: dict[str, None] = {}
    for node_id in _CITATION_PATTERN.findall(raw_text):
        seen.setdefault(node_id, None)
    return list(seen)


def build_prompt(query_text: str, nodes: list[NodeWithScore]) -> str:
    """Builds the user-role message: retrieved sources, then the question."""
    blocks = [
        f"[node:{node.node.node_id}]\n{node.node.get_content()}" for node in nodes
    ]
    sources = "\n\n".join(blocks) if blocks else "(no sources retrieved)"
    return f"Sources:\n{sources}\n\nQuestion: {query_text}"


class Answerer:
    def __init__(
        self,
        model: str | None = None,
        temperature: float = 0.0,
    ):
        routed_model = config.MODEL_ROUTING[_STAGE]["model"]
        # Model and temperature are fixed by config.MODEL_ROUTING and are
        # already applied when LLMFactory builds the client. These
        # parameters exist only to reject a contradicting value outright,
        # since a benchmark run that thinks it used a different model or
        # temperature would be unreproducible. The default is derived from
        # MODEL_ROUTING rather than a literal, so a future repoint of
        # "answerer" cannot silently go stale here.
        model = model or routed_model
        if model != routed_model:
            raise ValueError(
                f"Answerer model is fixed to {routed_model!r} by "
                f"config.MODEL_ROUTING[{_STAGE!r}]; got {model!r}"
            )
        if temperature != 0.0:
            raise ValueError(
                f"All benchmark LLM calls run at temperature=0; got {temperature!r}"
            )
        self.model = model
        self.temperature = temperature
        self._client = None

    def _get_client(self):
        # Built on first use, not in __init__, so constructing an Answerer
        # never reaches for API credentials, and the whole run reuses one
        # client (with its retry and concurrency wrapping) instead of
        # rebuilding it per call.
        if self._client is None:
            self._client = LLMFactory.get_client_for_stage(_STAGE)
        return self._client

    async def answer(self, query_text: str, nodes: list[NodeWithScore]) -> AnswerResult:
        client = self._get_client()
        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content=_SYSTEM_PROMPT),
            ChatMessage(role=MessageRole.USER, content=build_prompt(query_text, nodes)),
        ]

        start = time.monotonic()
        response = await client.achat(messages)
        latency_sec = time.monotonic() - start

        raw_text = response.message.content or ""
        usage = response.additional_kwargs or {}

        return AnswerResult(
            raw_text=raw_text,
            cited_node_ids=parse_citations(raw_text),
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            latency_sec=latency_sec,
        )


__all__ = ["AnswerResult", "Answerer", "build_prompt", "parse_citations"]
