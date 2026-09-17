"""Generator: proposes a query, ground truth answer and citations for one filing section.

Uses openai/gpt-oss-120b (config.MODEL_ROUTING["generator"]), a different
model family from the Critic, per the anti-self-grading invariant.
"""
import json

from llama_index.core.base.llms.types import ChatMessage, MessageRole

from llm_client.llm_factory import LLMFactory

_QUADRANT_GUIDANCE = {
    "Q1_Direct_Text": "Ask a direct fact-retrieval question answerable from a single explicit statement in continuous prose.",
    "Q2_Implicit_Text": "Ask a question requiring synthesis across multiple narrative passages in this section (not a single sentence).",
    "Q3_Direct_Table": "Ask a question requiring exact extraction of a specific cell/value from a table in this section.",
    "Q4_Implicit_Table": "Ask a question requiring a calculation or cross-row/footnote inference using a table in this section.",
}

_SYSTEM_PROMPT = (
    "You are generating one benchmark question from a section of a SEC 10-K filing. "
    "Respond with ONLY a JSON object (no markdown fences, no commentary): "
    '{"query_text": "...", "ground_truth_answer": "...", "gt_citations": ["node_id", ...]}. '
    "gt_citations must only contain node_id values that were given to you in the section."
)


async def generate_query(
    section: dict, quadrant: str, previous_attempt_feedback: str | None = None
) -> dict:
    guidance = _QUADRANT_GUIDANCE[quadrant]
    user_content = (
        f"{guidance}\n\n"
        f"Section: {section['section_header']}\n"
        f"Available node_ids: {section['node_ids']}\n\n"
        f"Section content:\n{section['content']}"
    )

    if previous_attempt_feedback:
        # temperature=0 means an identical prompt reproduces the identical
        # (already-rejected) output, so a retry must change the input:
        # state what failed and steer the Generator toward a genuinely
        # different candidate from the same section.
        user_content += (
            "\n\nA previous attempt for this section was rejected: "
            f"{previous_attempt_feedback}\n"
            "Propose a genuinely different question this time -- pick a "
            "different fact, number, or passage from the section content "
            "above, phrased so its citation and answer are unambiguous."
        )

    client = LLMFactory.get_client_for_stage("generator")

    # client is a LlamaIndex LLM (OpenAILike), not a raw OpenAI SDK client:
    # it exposes achat(messages), not chat.completions.create(...). Model
    # and temperature=0 are already fixed at construction, so they are not
    # passed again here.
    response = await client.achat([
        ChatMessage(role=MessageRole.SYSTEM, content=_SYSTEM_PROMPT),
        ChatMessage(role=MessageRole.USER, content=user_content),
    ])

    payload = json.loads(response.message.content)

    return {
        "query_text": payload["query_text"],
        "ground_truth_answer": payload["ground_truth_answer"],
        "gt_citations": payload["gt_citations"],
        "quadrant": quadrant,
        "document_id": section["document_id"],
    }