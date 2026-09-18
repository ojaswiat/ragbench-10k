"""Deterministic accept/reject logic for the Generator/Critic adversarial loop.

Plain code, not an LLM judgement. A candidate is accepted only if the
Critic's independent re-derivation agrees with the Generator's proposal on
citations, on the numbers in the answer and on the answer's overall meaning.
"""
import re

from fastembed import TextEmbedding
from core.model_cache import model_cache_dir

_NUMBER_RE = re.compile(r"-?\d[\d,]*\.?\d*")

# Embedding model reused from this project's existing P1/P3 retrieval choice,
# loaded via fastembed (ONNX Runtime): no Groq call, no GPU/torch dependency.
_EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# Cosine-similarity threshold for two answer texts being considered
# semantically matching. Chosen empirically: genuinely-restated matching
# facts scored 0.93-0.97 in manual testing, while clearly unrelated
# statements scored 0.45-0.69. 0.75 sits in the gap between the two,
# erring toward permissive since this gate's job is to catch unrelated
# text, not to arbitrate numeric correctness (that is values_match's job).
EMBEDDING_SIMILARITY_THRESHOLD = 0.75

_embedding_model: TextEmbedding | None = None


def _get_embedding_model() -> TextEmbedding:
    """Lazily construct and cache the embedding model (expensive to load)."""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = TextEmbedding(model_name=_EMBEDDING_MODEL_NAME, cache_dir=model_cache_dir())
    return _embedding_model


def citations_overlap(gt_citations: list[str], critic_cited_ids: list[str]) -> bool:
    return bool(set(gt_citations) & set(critic_cited_ids))


def extract_numbers(text: str) -> list[float]:
    numbers = []
    for match in _NUMBER_RE.findall(text or ""):
        cleaned = match.replace(",", "")
        try:
            numbers.append(float(cleaned))
        except ValueError:
            continue
    return numbers


def values_match(gt_answer: str, critic_answer: str, tolerance: float = 0.01) -> bool:
    """True if every number in gt_answer appears, within tolerance, somewhere
    in critic_answer's numbers. Extra numbers in critic_answer are allowed
    and ignored: this is a subset check, not an exact-count match.
    """
    gt_numbers = extract_numbers(gt_answer)
    critic_numbers = extract_numbers(critic_answer)

    if not gt_numbers or not critic_numbers:
        return gt_answer.strip().lower() == critic_answer.strip().lower()

    return all(
        any(abs(gt_number - critic_number) <= tolerance for critic_number in critic_numbers) for gt_number in gt_numbers
    )


def embedding_similarity(text_a: str, text_b: str) -> float:
    """Cosine similarity between the bge-small-en-v1.5 embeddings of two
    texts. Purely local ONNX inference: deterministic, no Groq call.
    """
    model = _get_embedding_model()
    emb_a, emb_b = list(model.embed([text_a or "", text_b or ""]))

    dot = sum(component_a * component_b for component_a, component_b in zip(emb_a, emb_b))
    norm_a = sum(component_a * component_a for component_a in emb_a) ** 0.5
    norm_b = sum(component_b * component_b for component_b in emb_b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


def embedding_similarity_ok(
    gt_answer: str, critic_answer: str, threshold: float = EMBEDDING_SIMILARITY_THRESHOLD
) -> bool:
    return embedding_similarity(gt_answer, critic_answer) >= threshold


def check_query(
    gt_citations: list[str],
    gt_answer: str,
    critic_cited_ids: list[str],
    critic_answer: str,
) -> bool:
    return (
        citations_overlap(gt_citations, critic_cited_ids)
        and values_match(gt_answer, critic_answer)
        and embedding_similarity_ok(gt_answer, critic_answer)
    )


def diagnose_rejection(
    gt_citations: list[str],
    gt_answer: str,
    critic_cited_ids: list[str],
    critic_answer: str,
) -> str:
    """Human-readable reason a rejected candidate failed check_query.

    Only meaningful after check_query has already returned False for the
    same inputs; used to build retry feedback for the Generator, not as
    part of the accept/reject decision itself.
    """
    if not citations_overlap(gt_citations, critic_cited_ids):
        return (
            f"citation mismatch: the Critic's independently-found node_ids "
            f"{critic_cited_ids} did not overlap the proposed gt_citations "
            f"{gt_citations}"
        )
    if not values_match(gt_answer, critic_answer):
        return (
            f"value mismatch: the Critic independently computed "
            f"'{critic_answer}', which does not match the proposed "
            f"ground_truth_answer '{gt_answer}'"
        )
    if not embedding_similarity_ok(gt_answer, critic_answer):
        similarity = embedding_similarity(gt_answer, critic_answer)
        return (
            f"embedding similarity too low: the Critic's answer "
            f"'{critic_answer}' scored {similarity:.3f} cosine similarity "
            f"against the proposed ground_truth_answer '{gt_answer}' "
            f"(threshold {EMBEDDING_SIMILARITY_THRESHOLD})"
        )
    return "no mismatch detected"
