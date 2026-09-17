from unittest.mock import AsyncMock, patch

import pytest

import pipelines.vector.build_vector_index as bvi
from pipelines.vector.p1_vector import P1VectorRetriever


async def _seed_collection(tmp_path, monkeypatch, nodes_by_document: dict[str, list[dict]]):
    # Seed via the real build path (build_index_for_document) rather than
    # hand-constructing TextNodes + VectorStoreIndex, so this fixture can't
    # drift from how the index is actually built in production.
    monkeypatch.setattr(bvi, "STORAGE_ROOT", tmp_path)
    for document_id, fake_nodes in nodes_by_document.items():
        with patch.object(bvi.dbm, "get_nodes_by_document", new=AsyncMock(return_value=fake_nodes)):
            await bvi.build_index_for_document(document_id)


def _fake_node(node_id, document_id, content, page_num=1):
    return {
        "node_id": node_id,
        "document_id": document_id,
        "parent_item_header": None,
        "node_type": "text",
        "source_page_num": page_num,
        "content": content,
        "token_count": len(content.split()),
    }


_AAPL_NODES = [
    _fake_node("AAPL_2025_n0001", "AAPL_2025", "Apple's total net sales were $394.3 billion in fiscal 2025.", 1),
    _fake_node("AAPL_2025_n0002", "AAPL_2025", "Apple's research and development expense grew 10 percent.", 2),
]
_MSFT_NODES = [
    _fake_node("MSFT_2025_n0001", "MSFT_2025", "Microsoft's total revenue was $245 billion in fiscal 2025.", 1),
]


@pytest.mark.asyncio
async def test_retrieve_filters_to_given_document_id(tmp_path, monkeypatch):
    await _seed_collection(tmp_path, monkeypatch, {"AAPL_2025": _AAPL_NODES, "MSFT_2025": _MSFT_NODES})

    retriever = P1VectorRetriever(storage_root=tmp_path)
    results = retriever.retrieve("What were total net sales?", document_id="AAPL_2025", k=5)

    assert len(results) > 0
    assert all(node.node.metadata["document_id"] == "AAPL_2025" for node in results)
    node_ids = {node.node.node_id for node in results}
    assert "MSFT_2025_n0001" not in node_ids


@pytest.mark.asyncio
async def test_retrieve_respects_k(tmp_path, monkeypatch):
    await _seed_collection(tmp_path, monkeypatch, {"AAPL_2025": _AAPL_NODES, "MSFT_2025": _MSFT_NODES})

    retriever = P1VectorRetriever(storage_root=tmp_path)
    results = retriever.retrieve("What were total net sales?", document_id="AAPL_2025", k=1)

    assert len(results) == 1


@pytest.mark.asyncio
async def test_retrieve_returns_most_relevant_node_first(tmp_path, monkeypatch):
    await _seed_collection(tmp_path, monkeypatch, {"AAPL_2025": _AAPL_NODES, "MSFT_2025": _MSFT_NODES})

    retriever = P1VectorRetriever(storage_root=tmp_path)
    results = retriever.retrieve("What were Apple's total net sales?", document_id="AAPL_2025", k=2)

    assert results[0].node.node_id == "AAPL_2025_n0001"


@pytest.mark.asyncio
async def test_retrieve_returns_empty_list_for_unindexed_document_id(tmp_path, monkeypatch):
    # An unindexed or misspelled document_id has no rows to filter to; this
    # is a normal "no results" case, not an error condition, so retrieve()
    # must return an empty list rather than raising.
    await _seed_collection(tmp_path, monkeypatch, {"AAPL_2025": _AAPL_NODES})

    retriever = P1VectorRetriever(storage_root=tmp_path)
    results = retriever.retrieve("What were total net sales?", document_id="NOT_A_REAL_DOC", k=5)

    assert results == []
