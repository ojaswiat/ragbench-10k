from unittest.mock import AsyncMock, patch

import pytest

import ingest.run_ingestion as run_ingestion


@pytest.mark.asyncio
async def test_ingest_one_calls_fetch_parse_build_and_insert(tmp_path):
    entry = {"document_id": "AAPL_2025", "ticker": "AAPL", "fiscal_year": 2025}

    with patch.object(run_ingestion.dbm, "get_nodes_by_document", new=AsyncMock(return_value=[])), \
         patch.object(run_ingestion, "fetch_filing", new=AsyncMock(return_value="/tmp/fake.html")), \
         patch.object(run_ingestion, "parse_filing", new=AsyncMock(return_value="/tmp/fake.md")), \
         patch("builtins.open", create=True) as mock_open, \
         patch.object(run_ingestion, "build_nodes", return_value=[
             {"node_id": "AAPL_2025_n0001", "document_id": "AAPL_2025", "parent_item_header": None,
              "node_type": "text", "source_page_num": None, "content": "x", "token_count": 1}
         ]) as mock_build, \
         patch.object(run_ingestion.dbm, "insert_node", new=AsyncMock()) as mock_insert:
        mock_open.return_value.__enter__.return_value.read.return_value = "markdown content"
        count = await run_ingestion.ingest_one(entry)

    assert count == 1
    mock_build.assert_called_once_with("AAPL_2025", "AAPL", 2025, "markdown content")
    mock_insert.assert_called_once()


@pytest.mark.asyncio
async def test_ingest_one_skips_already_ingested_document():
    entry = {"document_id": "AAPL_2025", "ticker": "AAPL", "fiscal_year": 2025}
    existing_nodes = [
        {"node_id": "AAPL_2025_n0001", "document_id": "AAPL_2025", "parent_item_header": None,
         "node_type": "text", "source_page_num": None, "content": "x", "token_count": 1},
        {"node_id": "AAPL_2025_n0002", "document_id": "AAPL_2025", "parent_item_header": None,
         "node_type": "text", "source_page_num": None, "content": "y", "token_count": 1},
    ]

    with patch.object(run_ingestion.dbm, "get_nodes_by_document", new=AsyncMock(return_value=existing_nodes)), \
         patch.object(run_ingestion, "fetch_filing", new=AsyncMock()) as mock_fetch, \
         patch.object(run_ingestion, "parse_filing", new=AsyncMock()) as mock_parse, \
         patch.object(run_ingestion.dbm, "insert_node", new=AsyncMock()) as mock_insert:
        count = await run_ingestion.ingest_one(entry)

    assert count == 2
    mock_fetch.assert_not_called()
    mock_parse.assert_not_called()
    mock_insert.assert_not_called()
