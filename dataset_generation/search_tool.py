"""Local, dependency-free keyword search the Critic uses to find evidence.

Not a retrieval pipeline: kept separate from the benchmarked BM25
pipeline. Scoring is a hand-rolled, length-normalised term frequency
rather than BM25 or TF-IDF, so it cannot bias which queries survive into
the dataset toward ones the real BM25 pipeline would score well on.
"""
import re

from pydantic import BaseModel, Field

_WORD_RE = re.compile(r"[a-z0-9]+")

# Floor for the length-normalisation denominator: plain division by node
# token length lets a tiny node (e.g. a one-token block like "Revenue")
# score close to 1.0 and beat a longer, substantive node containing the
# same term with real context. node_builder.py splits filing markdown on
# blank lines with no minimum block-length check, so real 10-K filings
# contain many such tiny blocks. 24 is the median token_count across the
# live corpus (18,297 nodes across the 13-filing manifest), so a node
# shorter than typical is scored as if it were typical-length rather than
# gaining an unfair advantage from being tiny.
_MIN_NODE_LENGTH_FOR_SCORING = 24

SEARCH_TOOL_NAME = "search_filing"

SEARCH_TOOL_DESCRIPTION = (
    "Search this filing's nodes for content matching a query. "
    "Returns up to 5 of the most relevant nodes with their node_id and content."
)


class SearchFilingArgs(BaseModel):
    """Argument schema for the search_filing tool offered to the Critic.

    A pydantic model, not a hand-written JSON blob, so SEARCH_TOOL_SCHEMA
    below, derived from it, can never drift out of sync.
    """

    query: str = Field(description="Keywords to search for in the filing")


SEARCH_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": SEARCH_TOOL_NAME,
        "description": SEARCH_TOOL_DESCRIPTION,
        # Mirrors LlamaIndex's own ToolMetadata.get_parameters_dict(): keep
        # only the JSON-Schema keys a provider's function definition accepts,
        # dropping pydantic's "title" and similar decoration.
        "parameters": {
            key: value
            for key, value in SearchFilingArgs.model_json_schema().items()
            if key in ("type", "properties", "required", "definitions", "$defs")
        },
    },
}


def _tokenize(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))


def search_filing_nodes(nodes: list[dict], query: str, top_k: int = 5) -> list[dict]:
    query_tokens = _tokenize(query)
    scored = []
    for node in nodes:
        node_tokens = _WORD_RE.findall(node["content"].lower())
        if not node_tokens:
            continue
        raw_count = sum(1 for token in node_tokens if token in query_tokens)
        if raw_count > 0:
            score = raw_count / max(len(node_tokens), _MIN_NODE_LENGTH_FOR_SCORING)
            scored.append((score, node["node_id"], node))
    scored.sort(key=lambda scored_entry: (-scored_entry[0], scored_entry[1]))
    return [node for _, _, node in scored[:top_k]]
