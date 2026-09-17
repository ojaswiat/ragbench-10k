from unittest.mock import AsyncMock, patch

import pytest
from llama_index.core.llms import MockLLM

import pipelines.structural.build_summary_index as bsi
import pipelines.structural.p3_structural as p3s
from pipelines.structural.p3_structural import P3StructuralRetriever


class _ForbiddenLLM(MockLLM):
    """Fails loudly if the query path ever calls an LLM."""

    def predict(self, *args, **kwargs):
        raise AssertionError("P3 retrieval must not call an LLM at query time")


async def _seed_index(tmp_path, monkeypatch, nodes_by_document: dict[str, list[dict]]):
    # Seed via the real build path (build_index_for_document) so the persisted
    # layout can't drift from production, with the summariser swapped for a
    # MockLLM: the tree build is the one LLM-calling stage, and the test
    # suite must never spend provider quota. Summary text is therefore
    # filler; only the leaf nodes carry real text, which retrieve() returns.
    monkeypatch.setattr(bsi, "STORAGE_ROOT", tmp_path)
    for document_id, fake_nodes in nodes_by_document.items():
        with patch.object(bsi.dbm, "get_nodes_by_document", new=AsyncMock(return_value=fake_nodes)), \
             patch.object(bsi.LLMFactory, "get_client_for_stage", return_value=MockLLM()):
            await bsi.build_index_for_document(document_id)


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
    _fake_node("AAPL_2025_n0003", "AAPL_2025", "Apple's headquarters are located in Cupertino California.", 3),
]
_MSFT_NODES = [
    _fake_node("MSFT_2025_n0001", "MSFT_2025", "Microsoft's total revenue was $245 billion in fiscal 2025.", 1),
]


@pytest.mark.asyncio
async def test_retrieve_filters_to_given_document_id(tmp_path, monkeypatch):
    await _seed_index(tmp_path, monkeypatch, {"AAPL_2025": _AAPL_NODES, "MSFT_2025": _MSFT_NODES})

    retriever = P3StructuralRetriever(storage_root=tmp_path)
    results = retriever.retrieve("What was total revenue?", document_id="AAPL_2025", k=3)

    assert len(results) > 0
    assert all(result.node.metadata["document_id"] == "AAPL_2025" for result in results)
    assert "MSFT_2025_n0001" not in {result.node.node_id for result in results}


@pytest.mark.asyncio
async def test_retrieve_respects_k(tmp_path, monkeypatch):
    await _seed_index(tmp_path, monkeypatch, {"AAPL_2025": _AAPL_NODES})

    retriever = P3StructuralRetriever(storage_root=tmp_path)

    assert len(retriever.retrieve("What were total net sales?", document_id="AAPL_2025", k=1)) == 1
    # k exceeding the filing's node count is a normal case: return what exists,
    # never more than k.
    assert len(retriever.retrieve("What were total net sales?", document_id="AAPL_2025", k=5)) <= 5


@pytest.mark.asyncio
async def test_retrieve_returns_most_relevant_node_first(tmp_path, monkeypatch):
    await _seed_index(tmp_path, monkeypatch, {"AAPL_2025": _AAPL_NODES})

    retriever = P3StructuralRetriever(storage_root=tmp_path)
    results = retriever.retrieve("What were Apple's total net sales?", document_id="AAPL_2025", k=3)

    assert results[0].node.node_id == "AAPL_2025_n0001"
    assert results[0].score is not None


@pytest.mark.asyncio
async def test_retrieve_returns_nodes_with_full_metadata(tmp_path, monkeypatch):
    await _seed_index(tmp_path, monkeypatch, {"AAPL_2025": _AAPL_NODES})

    retriever = P3StructuralRetriever(storage_root=tmp_path)
    results = retriever.retrieve("What were total net sales?", document_id="AAPL_2025", k=1)

    metadata = results[0].node.metadata
    assert set(metadata) >= {"document_id", "parent_item_header", "node_type", "source_page_num"}
    assert results[0].node.get_content()


@pytest.mark.asyncio
async def test_retrieve_makes_no_llm_call(tmp_path, monkeypatch):
    await _seed_index(tmp_path, monkeypatch, {"AAPL_2025": _AAPL_NODES})
    monkeypatch.setattr(p3s, "MockLLM", _ForbiddenLLM)

    retriever = P3StructuralRetriever(storage_root=tmp_path)
    results = retriever.retrieve("What were total net sales?", document_id="AAPL_2025", k=2)

    assert len(results) > 0


@pytest.mark.asyncio
async def test_retrieve_ranking_is_stable_across_calls(tmp_path, monkeypatch):
    await _seed_index(tmp_path, monkeypatch, {"AAPL_2025": _AAPL_NODES})

    retriever = P3StructuralRetriever(storage_root=tmp_path)
    first = retriever.retrieve("research and development expense", document_id="AAPL_2025", k=3)
    second = retriever.retrieve("research and development expense", document_id="AAPL_2025", k=3)

    assert [result.node.node_id for result in first] == [result.node.node_id for result in second]


def test_retrieve_raises_clearly_when_index_not_built(tmp_path):
    retriever = P3StructuralRetriever(storage_root=tmp_path)
    with pytest.raises(FileNotFoundError):
        retriever.retrieve("anything", document_id="NOT_BUILT", k=5)


def test_storage_root_defaults_to_build_module_constant(tmp_path, monkeypatch):
    monkeypatch.setattr(bsi, "STORAGE_ROOT", tmp_path)

    assert P3StructuralRetriever()._storage_root == tmp_path
