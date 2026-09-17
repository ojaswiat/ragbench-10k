"""Model routing, throttle flag, and env loading for the whole benchmark build."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

GROQ_API_ENDPOINT: str = os.getenv("GROQ_API_ENDPOINT", "https://api.groq.com/openai/v1")
NIM_API_ENDPOINT: str = os.getenv("NIM_API_ENDPOINT", "https://integrate.api.nvidia.com/v1")
OPENROUTER_API_ENDPOINT: str = os.getenv("OPENROUTER_API_ENDPOINT", "https://openrouter.ai/api/v1")

GROQ_API_KEY: str | None = os.getenv("GROQ_API_KEY")
LLAMA_CLOUD_API_KEY: str | None = os.getenv("LLAMA_CLOUD_API_KEY")
NIM_API_KEY: str | None = os.getenv("NIM_API_KEY")
OPENROUTER_API_KEY: str | None = os.getenv("OPENROUTER_API_KEY")
SEC_EDGAR_USER_AGENT: str = os.getenv("SEC_EDGAR_USER_AGENT", "rag-techniques-benchmark unknown@example.com")

LOCAL_TEST_THROTTLE: bool = os.getenv("LOCAL_TEST_THROTTLE", "true").lower() == "true"
THROTTLE_LIMIT: int = 3

# Reasoning-tuned models: without a cap, reasoning tokens exhaust the
# completion budget and leave message.content=None, crashing json.loads()
# in the caller. effort="none" is rejected by some endpoints; "low" is
# accepted everywhere tested and still bounds the spend.
_REASONING_LOW = {"reasoning": {"effort": "low"}}

# Single-upstream pinning. OpenRouter load-balances across hosts whose
# quantisation differs (fp8 vs bf16), so an unpinned run would mix
# numerically different backends across the 900 cells. allow_fallbacks=False
# makes a host outage fail loudly and resume later rather than silently
# switching backend mid-run.
_PIN_DEEPINFRA = {"provider": {"order": ["DeepInfra"], "allow_fallbacks": False}}
_PIN_CHUTES = {"provider": {"order": ["Chutes"], "allow_fallbacks": False}}

# Guardrails.md §2 — fixed model routing matrix. Do not change without updating the spec.
#
# Every OpenRouter stage carries _REASONING_LOW, even "answerer" whose Llama
# 3.3 70B model does not support the `reasoning` parameter. OpenRouter drops
# unsupported request fields rather than erroring, and sending the cap
# uniformly is what keeps the invariant simple and total instead of having
# to track which models happen to support it. Do not "clean up" by removing
# it from answerer.
MODEL_ROUTING: dict[str, dict] = {
    "generator":     {"model": "nvidia/nemotron-3-super-120b-a12b:free", "provider": "openrouter", "extra_body": _REASONING_LOW},
    "critic":        {"model": "openai/gpt-oss-20b:free", "provider": "openrouter", "extra_body": _REASONING_LOW},
    "p3_index_build":{"model": "nvidia/nemotron-3-super-120b-a12b", "provider": "nvidia"},
    "answerer":      {"model": "meta-llama/llama-3.3-70b-instruct", "provider": "openrouter", "extra_body": {**_PIN_DEEPINFRA, **_REASONING_LOW}},
    "judge":         {"model": "qwen/qwen3.6-27b", "provider": "openrouter", "extra_body": {**_PIN_CHUTES, **_REASONING_LOW}},
    "debug":         {"model": "llama-3.1-8b-instant",   "provider": "groq"},
}

GROQ_MAX_CONCURRENCY: int = 5
NIM_MAX_CONCURRENCY: int = 5
OPENROUTER_MAX_CONCURRENCY: int = 5