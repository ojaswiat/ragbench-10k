"""Converts nodes-table rows into LlamaIndex TextNode objects.

Kept separate from build_summary_index.py so the node-to-TextNode mapping
can be unit-tested without touching Groq.
"""
from llama_index.core.schema import TextNode


def nodes_to_llama_nodes(nodes: list[dict]) -> list[TextNode]:
    return [
        TextNode(
            id_=node["node_id"],
            text=node["content"],
            metadata={
                "document_id": node["document_id"],
                "parent_item_header": node["parent_item_header"],
                "node_type": node["node_type"],
                "source_page_num": node["source_page_num"],
            },
        )
        for node in nodes
    ]
