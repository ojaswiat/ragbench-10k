from unittest.mock import AsyncMock, patch

import pytest

import pipelines.vector.build_vector_index as bvi


@pytest.mark.asyncio
async def test_build_index_for_document_embeds_and_indexes_nodes(tmp_path, monkeypatch):
    monkeypatch.setattr(bvi, "STORAGE_ROOT", tmp_path)

    fake_nodes = [
        {"node_id": "AAPL_2025_n0001", "document_id": "AAPL_2025",
         "parent_item_header": None, "node_type": "text",
         "source_page_num": 1, "content": "Total net sales were $394.3 billion.",
         "token_count": 8},
        {"node_id": "AAPL_2025_n0002", "document_id": "AAPL_2025",
         "parent_item_header": None, "node_type": "text",
         "source_page_num": 2, "content": "Gross margin was 44 percent.",
         "token_count": 6},
    ]

    with patch.object(bvi.dbm, "get_nodes_by_document", new=AsyncMock(return_value=fake_nodes)):
        result = await bvi.build_index_for_document("AAPL_2025")

    assert result == {"document_id": "AAPL_2025", "skipped": False, "node_count": 2}
    assert bvi.is_document_indexed("AAPL_2025", storage_root=tmp_path) is True


@pytest.mark.asyncio
async def test_build_index_for_document_skips_if_already_indexed(tmp_path, monkeypatch):
    monkeypatch.setattr(bvi, "STORAGE_ROOT", tmp_path)

    fake_nodes = [
        {"node_id": "AAPL_2025_n0001", "document_id": "AAPL_2025",
         "parent_item_header": None, "node_type": "text",
         "source_page_num": 1, "content": "Total net sales were $394.3 billion.",
         "token_count": 8},
    ]

    with patch.object(bvi.dbm, "get_nodes_by_document", new=AsyncMock(return_value=fake_nodes)):
        first = await bvi.build_index_for_document("AAPL_2025")
        assert first["skipped"] is False

        second = await bvi.build_index_for_document("AAPL_2025")
        assert second == {"document_id": "AAPL_2025", "skipped": True, "node_count": 0}


@pytest.mark.asyncio
async def test_build_index_for_document_raises_on_no_nodes(tmp_path, monkeypatch):
    monkeypatch.setattr(bvi, "STORAGE_ROOT", tmp_path)

    with patch.object(bvi.dbm, "get_nodes_by_document", new=AsyncMock(return_value=[])):
        with pytest.raises(ValueError):
            await bvi.build_index_for_document("MSFT_2025")


def test_is_document_indexed_false_for_empty_collection(tmp_path):
    assert bvi.is_document_indexed("AAPL_2025", storage_root=tmp_path) is False


@pytest.mark.asyncio
async def test_build_index_for_document_isolates_nodes_across_documents(tmp_path, monkeypatch):
    # Node ids must not leak across documents sharing one collection.
    monkeypatch.setattr(bvi, "STORAGE_ROOT", tmp_path)

    nodes_a = [
        {"node_id": "AAPL_2025_n0001", "document_id": "AAPL_2025",
         "parent_item_header": None, "node_type": "text",
         "source_page_num": 1, "content": "Total net sales were $394.3 billion.",
         "token_count": 8},
        {"node_id": "AAPL_2025_n0002", "document_id": "AAPL_2025",
         "parent_item_header": None, "node_type": "text",
         "source_page_num": 2, "content": "Gross margin was 44 percent.",
         "token_count": 6},
    ]
    nodes_b = [
        {"node_id": "MSFT_2025_n0001", "document_id": "MSFT_2025",
         "parent_item_header": None, "node_type": "text",
         "source_page_num": 1, "content": "Revenue grew year over year.",
         "token_count": 6},
    ]

    with patch.object(bvi.dbm, "get_nodes_by_document", new=AsyncMock(return_value=nodes_a)):
        await bvi.build_index_for_document("AAPL_2025")

    with patch.object(bvi.dbm, "get_nodes_by_document", new=AsyncMock(return_value=nodes_b)):
        await bvi.build_index_for_document("MSFT_2025")

    collection = bvi.get_collection(tmp_path)

    doc_a_results = collection.get(where={"document_id": "AAPL_2025"})
    assert set(doc_a_results["ids"]) == {"AAPL_2025_n0001", "AAPL_2025_n0002"}

    doc_b_results = collection.get(where={"document_id": "MSFT_2025"})
    assert set(doc_b_results["ids"]) == {"MSFT_2025_n0001"}


@pytest.mark.asyncio
async def test_build_index_for_document_embeds_only_raw_content(tmp_path, monkeypatch):
    # parent_item_header, node_type, source_page_num and document_id live in
    # TextNode.metadata (set by the shared nodes_to_llama_nodes), but P1 must
    # embed only the filing text itself; anything else would leak structural
    # signal into the vector pipeline that P2/P3 don't get.
    monkeypatch.setattr(bvi, "STORAGE_ROOT", tmp_path)

    fake_nodes = [
        {"node_id": "AAPL_2025_n0001", "document_id": "AAPL_2025",
         "parent_item_header": "Item 7. MD&A", "node_type": "table",
         "source_page_num": 42, "content": "Total net sales were $394.3 billion.",
         "token_count": 8},
    ]

    with patch.object(bvi.dbm, "get_nodes_by_document", new=AsyncMock(return_value=fake_nodes)):
        await bvi.build_index_for_document("AAPL_2025")

    collection = bvi.get_collection(tmp_path)
    row = collection.get(ids=["AAPL_2025_n0001"], include=["documents"])
    assert row["documents"][0] == "Total net sales were $394.3 billion."
    assert "parent_item_header" not in row["documents"][0]
    assert "node_type" not in row["documents"][0]


@pytest.mark.asyncio
async def test_build_index_for_document_raises_on_colliding_ids_across_documents(tmp_path, monkeypatch):
    # Chroma's collection.add() keeps the OLD row and silently drops the NEW
    # one on a colliding id, without raising. Pre-seed a row under a node id
    # that a different document's build will reuse, then confirm the
    # post-index row-count check in build_index_for_document raises instead
    # of returning a false success.
    monkeypatch.setattr(bvi, "STORAGE_ROOT", tmp_path)

    stale_nodes = [
        {"node_id": "AAPL_2025_n0001", "document_id": "None",
         "parent_item_header": None, "node_type": "text",
         "source_page_num": 1, "content": "stale fixture row from an unrelated run.",
         "token_count": 6},
    ]
    with patch.object(bvi.dbm, "get_nodes_by_document", new=AsyncMock(return_value=stale_nodes)):
        await bvi.build_index_for_document("None")

    real_nodes = [
        {"node_id": "AAPL_2025_n0001", "document_id": "AAPL_2025",
         "parent_item_header": None, "node_type": "text",
         "source_page_num": 1, "content": "Total net sales were $394.3 billion.",
         "token_count": 8},
        {"node_id": "AAPL_2025_n0002", "document_id": "AAPL_2025",
         "parent_item_header": None, "node_type": "text",
         "source_page_num": 2, "content": "Gross margin was 44 percent.",
         "token_count": 6},
    ]
    with patch.object(bvi.dbm, "get_nodes_by_document", new=AsyncMock(return_value=real_nodes)):
        with pytest.raises(RuntimeError):
            await bvi.build_index_for_document("AAPL_2025")
