import os
import time
from unittest.mock import AsyncMock, patch

import pytest

import ingest.parse_filing as parse_filing


@pytest.mark.asyncio
async def test_parse_filing_calls_llamaparse_and_caches(tmp_path):
    raw_path = tmp_path / "AAPL_2025.html"
    raw_path.write_text("<html>fake filing</html>")
    parsed_dir = str(tmp_path / "parsed")

    with patch.object(
        parse_filing, "_parser"
    ) as mock_parser:
        mock_parser.aload_data = AsyncMock(return_value=[
            type("Doc", (), {"text": "# Item 1A. Risk Factors\n\nSome risk text."})()
        ])
        result_path = await parse_filing.parse_filing("AAPL_2025", str(raw_path), parsed_dir=parsed_dir)

    assert os.path.exists(result_path)
    with open(result_path) as handle:
        assert "Item 1A" in handle.read()
    mock_parser.aload_data.assert_called_once()


@pytest.mark.asyncio
async def test_parse_filing_skips_reparse_within_48h(tmp_path):
    raw_path = tmp_path / "AAPL_2025.html"
    raw_path.write_text("<html>fake filing</html>")
    parsed_dir = tmp_path / "parsed"
    parsed_dir.mkdir()
    cached_md = parsed_dir / "AAPL_2025.md"
    cached_md.write_text("already parsed")

    with patch.object(parse_filing, "_parser") as mock_parser:
        mock_parser.aload_data = AsyncMock()
        result_path = await parse_filing.parse_filing("AAPL_2025", str(raw_path), parsed_dir=str(parsed_dir))

    assert result_path == str(cached_md)
    mock_parser.aload_data.assert_not_called()


@pytest.mark.asyncio
async def test_parse_filing_reparses_if_cache_older_than_48h(tmp_path):
    raw_path = tmp_path / "AAPL_2025.html"
    raw_path.write_text("<html>fake filing</html>")
    parsed_dir = tmp_path / "parsed"
    parsed_dir.mkdir()
    cached_md = parsed_dir / "AAPL_2025.md"
    cached_md.write_text("stale parse")
    stale_time = time.time() - (49 * 3600)
    os.utime(cached_md, (stale_time, stale_time))

    with patch.object(parse_filing, "_parser") as mock_parser:
        mock_parser.aload_data = AsyncMock(return_value=[
            type("Doc", (), {"text": "fresh content"})()
        ])
        result_path = await parse_filing.parse_filing("AAPL_2025", str(raw_path), parsed_dir=str(parsed_dir))

    with open(result_path) as handle:
        assert handle.read() == "fresh content"
    mock_parser.aload_data.assert_called_once()
