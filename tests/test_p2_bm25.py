import asyncio
from unittest.mock import AsyncMock, patch

import pytest

import pipelines.bm25.build_bm25_index as bbi
from pipelines.bm25.p2_bm25 import P2BM25Retriever


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


def _seed_index(tmp_path, monkeypatch, nodes_by_document: dict[str, list[dict]]):
    # Sync on purpose: retrieve() below drives its own asyncio.run() call,
    # and nesting that inside a pytest-asyncio test coroutine's already
    # running loop raises RuntimeError. Seeding synchronously keeps every
    # test function plain sync, so retrieve()'s asyncio.run() always starts
    # from a clean, non-running loop, the same condition it runs under in
    # production.
    monkeypatch.setattr(bbi, "STORAGE_ROOT", tmp_path)
    for document_id, fake_nodes in nodes_by_document.items():
        with patch.object(bbi.dbm, "get_nodes_by_document", new=AsyncMock(return_value=fake_nodes)):
            asyncio.run(bbi.build_index_for_document(document_id))


# A 3rd node is required, not 2: with a 2-doc corpus a query term present in
# exactly 1 of 2 docs has BM25 IDF log((2-1+0.5)/(1+0.5)) == log(1) == 0
# exactly, so every AAPL query term below would score 0 across the board and
# "correct top-k" would pass only via the tie-break, not real ranking.
_AAPL_NODES = [
    _fake_node("AAPL_2025_n0001", "AAPL_2025", "Apple's total net sales were $394.3 billion in fiscal 2025.", 1),
    _fake_node("AAPL_2025_n0002", "AAPL_2025", "Apple's research and development expense grew 10 percent.", 2),
    _fake_node("AAPL_2025_n0003", "AAPL_2025", "Apple's headquarters are located in Cupertino California.", 3),
]
_MSFT_NODES = [
    _fake_node("MSFT_2025_n0001", "MSFT_2025", "Microsoft's total revenue was $245 billion in fiscal 2025.", 1),
]


def test_retrieve_returns_correct_top_k(tmp_path, monkeypatch):
    _seed_index(tmp_path, monkeypatch, {"AAPL_2025": _AAPL_NODES})

    with patch.object(bbi.dbm, "get_nodes_by_document", new=AsyncMock(return_value=_AAPL_NODES)):
        retriever = P2BM25Retriever(storage_root=tmp_path)
        results = retriever.retrieve("total net sales", document_id="AAPL_2025", k=1)

    assert len(results) == 1
    assert results[0].node.node_id == "AAPL_2025_n0001"


def test_retrieve_filters_to_given_document_id(tmp_path, monkeypatch):
    _seed_index(tmp_path, monkeypatch, {"AAPL_2025": _AAPL_NODES, "MSFT_2025": _MSFT_NODES})

    # The node lookup returns both filings' nodes, so document_id scoping
    # must come from retrieve() itself rather than a narrowed mock: if it
    # ever ranked against the wrong document's node ids, the MSFT node could
    # genuinely surface here.
    combined = _AAPL_NODES + _MSFT_NODES
    with patch.object(bbi.dbm, "get_nodes_by_document", new=AsyncMock(return_value=combined)):
        retriever = P2BM25Retriever(storage_root=tmp_path)
        results = retriever.retrieve("total revenue", document_id="AAPL_2025", k=5)

    # k=5 exceeds the 3-node AAPL corpus, so plain score-sorted top-k
    # returns all 3, including AAPL_2025_n0003, which has zero query-term
    # overlap and a score of 0. That is correct top-k behaviour; the actual
    # requirement here is document_id scoping (no MSFT node ever appears),
    # not relevance filtering.
    node_ids = {node.node.node_id for node in results}
    assert node_ids <= {"AAPL_2025_n0001", "AAPL_2025_n0002", "AAPL_2025_n0003"}
    assert "MSFT_2025_n0001" not in node_ids


def test_retrieve_tie_break_is_stable_across_calls(tmp_path, monkeypatch):
    tied_nodes = [
        _fake_node("DOC_n0001", "DOC", "alpha beta", 1),
        _fake_node("DOC_n0002", "DOC", "alpha beta", 2),
    ]
    _seed_index(tmp_path, monkeypatch, {"DOC": tied_nodes})

    with patch.object(bbi.dbm, "get_nodes_by_document", new=AsyncMock(return_value=tied_nodes)):
        retriever = P2BM25Retriever(storage_root=tmp_path)
        first = retriever.retrieve("alpha beta", document_id="DOC", k=2)
        second = retriever.retrieve("alpha beta", document_id="DOC", k=2)

    assert [node.node.node_id for node in first] == ["DOC_n0001", "DOC_n0002"]
    assert [node.node.node_id for node in first] == [node.node.node_id for node in second]


def test_retrieve_raises_clearly_when_index_not_built(tmp_path):
    retriever = P2BM25Retriever(storage_root=tmp_path)
    with pytest.raises(FileNotFoundError):
        retriever.retrieve("anything", document_id="NOT_BUILT", k=5)
