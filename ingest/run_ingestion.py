"""Orchestrates fetch -> parse -> node_builder -> database_manager.insert_node
for every filing in data/filings_manifest.json, honouring LOCAL_TEST_THROTTLE.
"""
import asyncio
import json

import database_manager as dbm
import loop_template
from ingest.fetch_filings import fetch_filing
from ingest.node_builder import build_nodes
from ingest.parse_filing import parse_filing

DB_PATH = "benchmark.db"
MANIFEST_PATH = "data/filings_manifest.json"


async def ingest_one(entry: dict) -> int:
    # Any existing row count is treated as "done", so a crash mid-document
    # leaves a partial node set that this will not retry. Delete that
    # document's rows from `nodes` to force re-ingestion.
    existing = await dbm.get_nodes_by_document(DB_PATH, entry["document_id"])
    if existing:
        return len(existing)

    raw_path = await fetch_filing(entry["ticker"], entry["fiscal_year"], entry["document_id"])
    parsed_path = await parse_filing(entry["document_id"], raw_path)
    with open(parsed_path) as handle:
        markdown_text = handle.read()
    nodes = build_nodes(entry["document_id"], entry["ticker"], entry["fiscal_year"], markdown_text)
    for node in nodes:
        await dbm.insert_node(DB_PATH, node)
    return len(nodes)


async def main():
    await dbm.init_db(DB_PATH)
    with open(MANIFEST_PATH) as handle:
        manifest = json.load(handle)

    manifest = loop_template.apply_throttle(manifest)

    for entry in manifest:
        count = await ingest_one(entry)
        print(f"{entry['document_id']}: {count} nodes")


if __name__ == "__main__":
    asyncio.run(main())
