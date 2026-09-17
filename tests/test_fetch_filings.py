import os
from unittest.mock import MagicMock, patch

import pytest

import ingest.fetch_filings as fetch_filings

FAKE_TICKERS_JSON = {
    "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
    "1": {"cik_str": 789019, "ticker": "MSFT", "title": "Microsoft Corp"},
}

FAKE_SUBMISSIONS_JSON = {
    "filings": {
        "recent": {
            "form": ["10-K", "8-K", "10-K"],
            "reportDate": ["2025-09-28", "2025-06-01", "2024-09-30"],
            "accessionNumber": ["0000320193-25-000010", "0000320193-25-000005", "0000320193-24-000010"],
            "primaryDocument": ["aapl-20250928.htm", "aapl-8k.htm", "aapl-20240930.htm"],
        }
    }
}


def test_resolve_cik_finds_ticker():
    with patch.object(fetch_filings, "_get_json", return_value=FAKE_TICKERS_JSON) as mock_get:
        cik = fetch_filings.resolve_cik("AAPL")
    assert cik == "0000320193"
    mock_get.assert_called_once()


def test_resolve_cik_raises_for_unknown_ticker():
    with patch.object(fetch_filings, "_get_json", return_value=FAKE_TICKERS_JSON):
        with pytest.raises(ValueError, match="no CIK found for ticker"):
            fetch_filings.resolve_cik("NOPE")


@pytest.mark.asyncio
async def test_fetch_filing_downloads_and_caches(tmp_path):
    raw_dir = str(tmp_path)
    fake_html = b"<html>fake 10-K content</html>"

    with patch.object(fetch_filings, "resolve_cik", return_value="0000320193"), \
         patch.object(fetch_filings, "_get_json", return_value=FAKE_SUBMISSIONS_JSON), \
         patch.object(fetch_filings, "_download_bytes", return_value=fake_html) as mock_download:
        path = await fetch_filings.fetch_filing("AAPL", 2025, "AAPL_2025", raw_dir=raw_dir)

    assert os.path.exists(path)
    with open(path, "rb") as handle:
        assert handle.read() == fake_html
    mock_download.assert_called_once()


@pytest.mark.asyncio
async def test_fetch_filing_skips_download_if_cached(tmp_path):
    raw_dir = str(tmp_path)
    cached_path = os.path.join(raw_dir, "AAPL_2025.html")
    with open(cached_path, "wb") as handle:
        handle.write(b"already cached")

    with patch.object(fetch_filings, "resolve_cik") as mock_resolve, \
         patch.object(fetch_filings, "_download_bytes") as mock_download:
        path = await fetch_filings.fetch_filing("AAPL", 2025, "AAPL_2025", raw_dir=raw_dir)

    assert path == cached_path
    mock_resolve.assert_not_called()
    mock_download.assert_not_called()


@pytest.mark.asyncio
async def test_fetch_filing_falls_back_to_paginated_archives(tmp_path):
    raw_dir = str(tmp_path)
    fake_html = b"<html>fake older 10-K</html>"

    recent_no_match = {
        "filings": {
            "recent": {"form": ["8-K"], "reportDate": ["2025-12-31"], "accessionNumber": ["0"], "primaryDocument": ["x.htm"]},
            "files": [{"name": "CIK0000019617-submissions-001.json"}],
        }
    }
    paginated_with_match = {
        "form": ["10-K"],
        "reportDate": ["2023-12-31"],
        "accessionNumber": ["0000019617-24-000010"],
        "primaryDocument": ["jpm-20231231.htm"],
    }

    with patch.object(fetch_filings, "resolve_cik", return_value="0000019617"), \
         patch.object(fetch_filings, "_get_json", side_effect=[recent_no_match, paginated_with_match]) as mock_get, \
         patch.object(fetch_filings, "_download_bytes", return_value=fake_html) as mock_download:
        path = await fetch_filings.fetch_filing("JPM", 2023, "JPM_2023", raw_dir=raw_dir)

    assert os.path.exists(path)
    with open(path, "rb") as handle:
        assert handle.read() == fake_html
    assert mock_get.call_count == 2
    mock_download.assert_called_once()


@pytest.mark.asyncio
async def test_fetch_filing_raises_when_no_matching_10k(tmp_path):
    raw_dir = str(tmp_path)
    empty_submissions = {"filings": {"recent": {"form": [], "reportDate": [], "accessionNumber": [], "primaryDocument": []}}}
    with patch.object(fetch_filings, "resolve_cik", return_value="0000320193"), \
         patch.object(fetch_filings, "_get_json", return_value=empty_submissions):
        with pytest.raises(ValueError, match="no 10-K found"):
            await fetch_filings.fetch_filing("AAPL", 2025, "AAPL_2025", raw_dir=raw_dir)
