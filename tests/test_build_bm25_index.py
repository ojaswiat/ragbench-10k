import pickle
from unittest.mock import AsyncMock, patch

import pytest

import pipelines.bm25.build_bm25_index as bbi


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


@pytest.mark.asyncio
async def test_build_index_for_document_builds_and_pickles_corpus(tmp_path, monkeypatch):
    monkeypatch.setattr(bbi, "STORAGE_ROOT", tmp_path)

    fake_nodes = [
        _fake_node("AAPL_2025_n0001", "AAPL_2025", "Total net sales were $394.3 billion."),
        _fake_node("AAPL_2025_n0002", "AAPL_2025", "Gross margin was 44 percent.", 2),
        _fake_node("AAPL_2025_n0003", "AAPL_2025", "Operating expenses increased slightly.", 3),
    ]

    with patch.object(bbi.dbm, "get_nodes_by_document", new=AsyncMock(return_value=fake_nodes)):
        result = await bbi.build_index_for_document("AAPL_2025")

    assert result == {"document_id": "AAPL_2025", "skipped": False, "node_count": 3}
    assert bbi.is_document_indexed("AAPL_2025", storage_root=tmp_path) is True

    with open(tmp_path / "AAPL_2025.pkl", "rb") as handle:
        payload = pickle.load(handle)
    assert payload["node_ids"] == ["AAPL_2025_n0001", "AAPL_2025_n0002", "AAPL_2025_n0003"]
    assert payload["bm25"].get_scores(["sales"])[0] > 0


@pytest.mark.asyncio
async def test_build_index_for_document_skips_if_already_indexed(tmp_path, monkeypatch):
    monkeypatch.setattr(bbi, "STORAGE_ROOT", tmp_path)

    fake_nodes = [_fake_node("AAPL_2025_n0001", "AAPL_2025", "Total net sales were $394.3 billion.")]

    with patch.object(bbi.dbm, "get_nodes_by_document", new=AsyncMock(return_value=fake_nodes)):
        first = await bbi.build_index_for_document("AAPL_2025")
        assert first["skipped"] is False

        second = await bbi.build_index_for_document("AAPL_2025")
        assert second == {"document_id": "AAPL_2025", "skipped": True, "node_count": 0}


@pytest.mark.asyncio
async def test_build_index_for_document_raises_on_no_nodes(tmp_path, monkeypatch):
    monkeypatch.setattr(bbi, "STORAGE_ROOT", tmp_path)

    with patch.object(bbi.dbm, "get_nodes_by_document", new=AsyncMock(return_value=[])):
        with pytest.raises(ValueError):
            await bbi.build_index_for_document("MSFT_2025")


def test_is_document_indexed_false_for_empty_storage(tmp_path):
    assert bbi.is_document_indexed("AAPL_2025", storage_root=tmp_path) is False


@pytest.mark.asyncio
async def test_build_index_for_document_isolates_nodes_across_documents(tmp_path, monkeypatch):
    monkeypatch.setattr(bbi, "STORAGE_ROOT", tmp_path)

    nodes_a = [_fake_node("AAPL_2025_n0001", "AAPL_2025", "Total net sales were $394.3 billion.")]
    nodes_b = [_fake_node("MSFT_2025_n0001", "MSFT_2025", "Revenue grew year over year.")]

    with patch.object(bbi.dbm, "get_nodes_by_document", new=AsyncMock(return_value=nodes_a)):
        await bbi.build_index_for_document("AAPL_2025")
    with patch.object(bbi.dbm, "get_nodes_by_document", new=AsyncMock(return_value=nodes_b)):
        await bbi.build_index_for_document("MSFT_2025")

    with open(tmp_path / "AAPL_2025.pkl", "rb") as handle:
        payload_a = pickle.load(handle)
    with open(tmp_path / "MSFT_2025.pkl", "rb") as handle:
        payload_b = pickle.load(handle)

    assert payload_a["node_ids"] == ["AAPL_2025_n0001"]
    assert payload_b["node_ids"] == ["MSFT_2025_n0001"]


@pytest.mark.asyncio
async def test_build_index_for_document_atomic_write_survives_interruption(tmp_path, monkeypatch):
    monkeypatch.setattr(bbi, "STORAGE_ROOT", tmp_path)

    def _raise(*args, **kwargs):
        raise RuntimeError("simulated crash")

    monkeypatch.setattr(bbi.pickle, "dump", _raise)

    fake_nodes = [_fake_node("AAPL_2025_n0001", "AAPL_2025", "Total net sales were $394.3 billion.")]

    with patch.object(bbi.dbm, "get_nodes_by_document", new=AsyncMock(return_value=fake_nodes)):
        with pytest.raises(RuntimeError):
            await bbi.build_index_for_document("AAPL_2025")

    assert not (tmp_path / "AAPL_2025.pkl").exists()
    assert (tmp_path / "AAPL_2025.pkl.tmp").exists()
