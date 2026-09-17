"""Groups filing nodes into per-section chunks for the Generator.

A "section" is every node sharing the same (document_id, parent_item_header),
case-normalised. Nodes with no header (cover pages, tables of contents,
signature blocks, roughly 6% of the corpus) are excluded.
"""


def group_sections(nodes: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str], dict] = {}

    for node in nodes:
        header = node.get("parent_item_header")
        if not header or not header.strip():
            continue

        key = (node["document_id"], header.strip().upper())
        if key not in groups:
            groups[key] = {
                "document_id": node["document_id"],
                "section_header": header.strip(),
                "node_ids": [],
                "content_parts": [],
                "token_count": 0,
            }

        groups[key]["node_ids"].append(node["node_id"])
        groups[key]["content_parts"].append(node["content"])
        groups[key]["token_count"] += node["token_count"]

    return [
        {
            "document_id": group["document_id"],
            "section_header": group["section_header"],
            "node_ids": group["node_ids"],
            "content": "\n\n".join(group["content_parts"]),
            "token_count": group["token_count"],
        }
        for group in groups.values()
    ]
