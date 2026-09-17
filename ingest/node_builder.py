"""Splits parsed Markdown into atomic TextNode-shaped dicts.

A table is kept as one node, never split. Each node keeps the nearest
preceding section header; source_page_num is always None.
"""
import re

_HEADER_RE = re.compile(r"^#{1,3}\s*(Item\s+\d+[A-Za-z]?\.?\s*.+?)\s*$", re.IGNORECASE)


def _is_table_block(block: str) -> bool:
    lines = [line for line in block.splitlines() if line.strip()]
    if not lines:
        return False
    # A caption line before the header row (e.g. an exhibit index) means the
    # first line alone cannot identify a table, so this checks the whole
    # block for at least two pipe-led lines: a header row plus a data or
    # separator row.
    table_lines = sum(1 for line in lines if line.strip().startswith("|"))
    return table_lines >= 2


def build_nodes(document_id: str, ticker: str, fiscal_year: int, markdown_text: str) -> list[dict]:
    blocks = [candidate_block for candidate_block in markdown_text.split("\n\n") if candidate_block.strip()]

    nodes = []
    current_header = None
    counter = 1

    for block in blocks:
        stripped = block.strip()
        header_match = _HEADER_RE.match(stripped)
        if header_match:
            current_header = header_match.group(1).strip()
            continue

        node_type = "table" if _is_table_block(stripped) else "text"
        node_id = f"{document_id}_n{counter:04d}"
        nodes.append({
            "node_id": node_id,
            "document_id": document_id,
            "parent_item_header": current_header,
            "node_type": node_type,
            "source_page_num": None,
            "content": stripped,
            "token_count": len(stripped.split()),
        })
        counter += 1

    return nodes
