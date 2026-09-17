import pytest

import database_manager as dbm
import ingest.parsing_audit as parsing_audit

TEST_DB = "test_audit.db"


@pytest.fixture(autouse=True)
def clean_db():
    import os
    for suffix in ("", "-wal", "-shm"):
        if os.path.exists(TEST_DB + suffix):
            os.remove(TEST_DB + suffix)
    yield
    for suffix in ("", "-wal", "-shm"):
        if os.path.exists(TEST_DB + suffix):
            os.remove(TEST_DB + suffix)


@pytest.mark.asyncio
async def test_sample_sections_returns_at_most_n():
    await dbm.init_db(TEST_DB)
    for index in range(5):
        await dbm.insert_node(TEST_DB, {
            "node_id": f"TEST_2025_n{index:04d}", "document_id": "TEST_2025",
            "parent_item_header": "Item 1A", "node_type": "text",
            "source_page_num": None, "content": f"content {index}", "token_count": 2,
        })
    samples = await parsing_audit.sample_sections(TEST_DB, sample_size=20)
    assert len(samples) == 5


@pytest.mark.asyncio
async def test_sample_sections_caps_at_n():
    await dbm.init_db(TEST_DB)
    for index in range(30):
        await dbm.insert_node(TEST_DB, {
            "node_id": f"TEST_2025_n{index:04d}", "document_id": "TEST_2025",
            "parent_item_header": "Item 1A", "node_type": "text",
            "source_page_num": None, "content": f"content {index}", "token_count": 2,
        })
    samples = await parsing_audit.sample_sections(TEST_DB, sample_size=20)
    assert len(samples) == 20


def test_write_audit_report_creates_readable_markdown(tmp_path):
    samples = [
        {"node_id": "TEST_2025_n0001", "document_id": "TEST_2025", "parent_item_header": "Item 1A",
         "node_type": "text", "content": "Sample risk text.", "token_count": 3},
    ]
    out_path = str(tmp_path / "audit_report.md")
    parsing_audit.write_audit_report(samples, out_path)

    with open(out_path) as handle:
        text = handle.read()
    assert "TEST_2025_n0001" in text
    assert "Item 1A" in text
    assert "Sample risk text." in text
