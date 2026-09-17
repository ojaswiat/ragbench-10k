from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import json

import pytest

import pipelines.structural.build_summary_index as bsi


@pytest.mark.asyncio
async def test_build_index_for_document_skips_if_final_dir_exists(tmp_path, monkeypatch):
    monkeypatch.setattr(bsi, "STORAGE_ROOT", tmp_path)
    (tmp_path / "AAPL_2025").mkdir()

    result = await bsi.build_index_for_document("AAPL_2025")

    assert result == {"document_id": "AAPL_2025", "skipped": True}


@pytest.mark.asyncio
async def test_build_index_for_document_builds_and_persists(tmp_path, monkeypatch):
    monkeypatch.setattr(bsi, "STORAGE_ROOT", tmp_path)

    fake_nodes = [{"node_id": "AAPL_2025_n0001", "document_id": "AAPL_2025",
                   "parent_item_header": None, "node_type": "text",
                   "source_page_num": None, "content": "hello", "token_count": 1}]

    fake_index = MagicMock()

    def fake_persist(persist_dir):
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        (Path(persist_dir) / "docstore.json").write_text("{}")

    fake_index.storage_context.persist.side_effect = fake_persist

    with patch.object(bsi.dbm, "get_nodes_by_document", new=AsyncMock(return_value=fake_nodes)), \
         patch.object(bsi, "TreeIndex", return_value=fake_index) as mock_tree, \
         patch.object(bsi.LLMFactory, "get_client_for_stage"):
        result = await bsi.build_index_for_document("AAPL_2025")

    assert result["document_id"] == "AAPL_2025"
    assert result["skipped"] is False
    assert (tmp_path / "AAPL_2025" / "docstore.json").exists()
    assert not (tmp_path / "AAPL_2025.tmp").exists()
    mock_tree.assert_called_once()


@pytest.mark.asyncio
async def test_build_index_for_document_crash_leaves_only_temp_dir(tmp_path, monkeypatch):
    """A crash between persist() and the atomic rename must never leave a
    false-positive cache hit; the final dir must stay absent."""
    monkeypatch.setattr(bsi, "STORAGE_ROOT", tmp_path)

    fake_nodes = [{"node_id": "AAPL_2025_n0001", "document_id": "AAPL_2025",
                   "parent_item_header": None, "node_type": "text",
                   "source_page_num": None, "content": "hello", "token_count": 1}]

    fake_index = MagicMock()

    def fake_persist(persist_dir):
        Path(persist_dir).mkdir(parents=True, exist_ok=True)

    fake_index.storage_context.persist.side_effect = fake_persist

    with patch.object(bsi.dbm, "get_nodes_by_document", new=AsyncMock(return_value=fake_nodes)), \
         patch.object(bsi, "TreeIndex", return_value=fake_index), \
         patch.object(bsi.LLMFactory, "get_client_for_stage"), \
         patch.object(bsi.os, "rename", side_effect=OSError("simulated crash")):
        with pytest.raises(OSError):
            await bsi.build_index_for_document("AAPL_2025")

    assert (tmp_path / "AAPL_2025.tmp").exists()
    assert not (tmp_path / "AAPL_2025").exists()
    assert bsi.is_built("AAPL_2025") is False


@pytest.mark.asyncio
async def test_build_index_for_document_cleans_stale_temp_dir_before_retry(tmp_path, monkeypatch):
    """A leftover temp dir from a prior crashed build must not break the next attempt."""
    monkeypatch.setattr(bsi, "STORAGE_ROOT", tmp_path)
    stale_temp = tmp_path / "AAPL_2025.tmp"
    stale_temp.mkdir()
    (stale_temp / "leftover.json").write_text("{}")

    fake_nodes = [{"node_id": "AAPL_2025_n0001", "document_id": "AAPL_2025",
                   "parent_item_header": None, "node_type": "text",
                   "source_page_num": None, "content": "hello", "token_count": 1}]

    fake_index = MagicMock()

    def fake_persist(persist_dir):
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        (Path(persist_dir) / "docstore.json").write_text("{}")

    fake_index.storage_context.persist.side_effect = fake_persist

    with patch.object(bsi.dbm, "get_nodes_by_document", new=AsyncMock(return_value=fake_nodes)), \
         patch.object(bsi, "TreeIndex", return_value=fake_index), \
         patch.object(bsi.LLMFactory, "get_client_for_stage"):
        result = await bsi.build_index_for_document("AAPL_2025")

    assert result["skipped"] is False
    assert (tmp_path / "AAPL_2025" / "docstore.json").exists()
    assert not (tmp_path / "AAPL_2025.tmp").exists()


@pytest.mark.asyncio
async def test_build_index_for_document_raises_on_no_nodes(tmp_path, monkeypatch):
    """A document_id with zero ingested nodes must fail loudly, not silently
    build/persist an empty TreeIndex (which would create a permanent
    false-positive cache hit)."""
    monkeypatch.setattr(bsi, "STORAGE_ROOT", tmp_path)

    with patch.object(bsi.dbm, "get_nodes_by_document", new=AsyncMock(return_value=[])), \
         patch.object(bsi, "TreeIndex") as mock_tree, \
         patch.object(bsi.LLMFactory, "get_client_for_stage"):
        with pytest.raises(ValueError):
            await bsi.build_index_for_document("MSFT_2025")

    mock_tree.assert_not_called()
    assert bsi.is_built("MSFT_2025") is False
    assert not (tmp_path / "MSFT_2025").exists()
    assert not (tmp_path / "MSFT_2025.tmp").exists()


@pytest.mark.asyncio
async def test_build_index_for_document_shares_callback_manager_with_llm_and_tree(tmp_path, monkeypatch):
    """The same CallbackManager must reach both LLMFactory.get_client_for_stage()
    and TreeIndex(): llama_index's llm_chat_callback() decorator fires
    token-usage events into the LLM's own callback_manager, not the index's,
    so the two must share one for token counting to work."""
    monkeypatch.setattr(bsi, "STORAGE_ROOT", tmp_path)

    fake_nodes = [{"node_id": "AAPL_2025_n0001", "document_id": "AAPL_2025",
                   "parent_item_header": None, "node_type": "text",
                   "source_page_num": None, "content": "hello", "token_count": 1}]

    fake_index = MagicMock()

    def fake_persist(persist_dir):
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        (Path(persist_dir) / "docstore.json").write_text("{}")

    fake_index.storage_context.persist.side_effect = fake_persist

    with patch.object(bsi.dbm, "get_nodes_by_document", new=AsyncMock(return_value=fake_nodes)), \
         patch.object(bsi, "TreeIndex", return_value=fake_index) as mock_tree, \
         patch.object(bsi.LLMFactory, "get_client_for_stage") as mock_get_client:
        await bsi.build_index_for_document("AAPL_2025")

    mock_get_client.assert_called_once()
    _, get_client_kwargs = mock_get_client.call_args
    assert get_client_kwargs["callback_manager"] is not None

    _, tree_kwargs = mock_tree.call_args
    assert tree_kwargs["callback_manager"] is get_client_kwargs["callback_manager"]


def test_confirm_build_prints_node_count_and_accepts_y(monkeypatch, capsys):
    monkeypatch.setattr("builtins.input", lambda _: "y")
    assert bsi.confirm_build("JPM_2025", 42) is True
    assert "JPM_2025 contains 42 nodes" in capsys.readouterr().out


def test_confirm_build_rejects_n(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "n")
    assert bsi.confirm_build("JPM_2025", 42) is False


def test_confirm_build_reprompts_on_invalid_input(monkeypatch):
    answers = iter(["maybe", "y"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    assert bsi.confirm_build("JPM_2025", 42) is True


@pytest.mark.asyncio
async def test_main_skips_document_when_user_declines(tmp_path, monkeypatch):
    manifest = [{"document_id": "JPM_2025", "ticker": "JPM", "fiscal_year": 2025}]
    monkeypatch.setattr(bsi, "STORAGE_ROOT", tmp_path)
    monkeypatch.setattr(bsi.loop_template, "apply_throttle", lambda items: items)

    with patch("builtins.open", create=True) as mock_open, \
         patch.object(bsi.dbm, "get_nodes_by_document", new=AsyncMock(return_value=[{"node_id": "n1"}])), \
         patch.object(bsi, "confirm_build", return_value=False), \
         patch.object(bsi, "build_index_for_document", new=AsyncMock()) as mock_build, \
         patch.object(bsi, "append_cost_log") as mock_log:
        mock_open.return_value.__enter__.return_value.read.return_value = json.dumps(manifest)
        await bsi.main()

    mock_build.assert_not_called()
    mock_log.assert_not_called()


def test_append_cost_log_creates_and_appends(tmp_path):
    log_path = tmp_path / "logs" / "index_build_costs.json"
    bsi.append_cost_log(
        {"document_id": "AAPL_2025", "skipped": False, "wall_clock_sec": 1.0,
         "input_tokens": 10, "output_tokens": 5},
        log_path=log_path,
    )
    bsi.append_cost_log({"document_id": "AAPL_2024", "skipped": True}, log_path=log_path)

    rows = json.loads(log_path.read_text())
    assert len(rows) == 2
    assert rows[0]["document_id"] == "AAPL_2025"
    assert rows[1]["skipped"] is True


@pytest.mark.asyncio
async def test_main_builds_each_manifest_entry_sequentially(tmp_path, monkeypatch):
    manifest = [
        {"document_id": "AAPL_2023", "ticker": "AAPL", "fiscal_year": 2023},
        {"document_id": "AAPL_2024", "ticker": "AAPL", "fiscal_year": 2024},
    ]
    monkeypatch.setattr(bsi, "STORAGE_ROOT", tmp_path)
    monkeypatch.setattr(bsi.loop_template, "apply_throttle", lambda items: items)

    with patch("builtins.open", create=True) as mock_open, \
         patch.object(bsi.dbm, "get_nodes_by_document", new=AsyncMock(return_value=[{"node_id": "n1"}])), \
         patch.object(bsi, "confirm_build", return_value=True) as mock_confirm, \
         patch.object(bsi, "build_index_for_document", new=AsyncMock(
             side_effect=[{"document_id": "AAPL_2023", "skipped": False},
                          {"document_id": "AAPL_2024", "skipped": True}])) as mock_build, \
         patch.object(bsi, "append_cost_log") as mock_log:
        mock_open.return_value.__enter__.return_value.read.return_value = json.dumps(manifest)
        await bsi.main()

    assert mock_confirm.call_count == 2
    mock_confirm.assert_any_call("AAPL_2023", 1)
    mock_confirm.assert_any_call("AAPL_2024", 1)

    assert mock_build.call_count == 2
    mock_build.assert_any_call("AAPL_2023")
    mock_build.assert_any_call("AAPL_2024")
    assert mock_log.call_count == 2
