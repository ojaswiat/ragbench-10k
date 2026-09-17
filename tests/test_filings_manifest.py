import json
from pathlib import Path

MANIFEST_PATH = Path(__file__).parent.parent / "data" / "filings_manifest.json"


def test_manifest_exists_and_is_a_list():
    assert MANIFEST_PATH.exists()
    manifest = json.loads(MANIFEST_PATH.read_text())
    assert isinstance(manifest, list)
    assert len(manifest) > 0


def test_every_entry_has_required_keys():
    manifest = json.loads(MANIFEST_PATH.read_text())
    for entry in manifest:
        assert set(entry.keys()) == {"document_id", "ticker", "fiscal_year"}
        assert entry["document_id"] == f"{entry['ticker']}_{entry['fiscal_year']}"


def test_document_ids_are_unique():
    manifest = json.loads(MANIFEST_PATH.read_text())
    ids = [entry["document_id"] for entry in manifest]
    assert len(ids) == len(set(ids))
