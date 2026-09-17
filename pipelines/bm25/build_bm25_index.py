"""Builds the per-document P2 BM25 index: one pickled BM25Okapi corpus
per filing under storage/bm25/.

Purely statistical, rank_bm25 over the shared tokenizer, no embeddings
and no LLM call. Resumable at document granularity.
"""
import asyncio
import json
import os
import pickle
from pathlib import Path

from rank_bm25 import BM25Okapi

import database_manager as dbm
import loop_template
from pipelines.bm25.tokenizer import tokenize

STORAGE_ROOT = Path("storage/bm25")
DB_PATH = "benchmark.db"
MANIFEST_PATH = "data/filings_manifest.json"


def index_path(document_id: str, storage_root: Path) -> Path:
    return storage_root / f"{document_id}.pkl"


def is_document_indexed(document_id: str, storage_root: Path | None = None) -> bool:
    if storage_root is None:
        storage_root = STORAGE_ROOT
    return index_path(document_id, storage_root).exists()


async def build_index_for_document(
    document_id: str,
    db_path: str = DB_PATH,
    storage_root: Path | None = None,
) -> dict:
    if storage_root is None:
        storage_root = STORAGE_ROOT

    if is_document_indexed(document_id, storage_root=storage_root):
        return {"document_id": document_id, "skipped": True, "node_count": 0}

    nodes = await dbm.get_nodes_by_document(db_path, document_id)
    if not nodes:
        raise ValueError(
            f"No nodes found for document_id={document_id!r}. Run Phase 2 "
            "ingestion for this filing before building the BM25 index."
        )

    tokenized_corpus = [tokenize(node["content"]) for node in nodes]
    bm25 = BM25Okapi(tokenized_corpus)
    payload = {"node_ids": [node["node_id"] for node in nodes], "bm25": bm25}

    storage_root.mkdir(parents=True, exist_ok=True)
    final_path = index_path(document_id, storage_root)
    temp_path = storage_root / f"{document_id}.pkl.tmp"
    with open(temp_path, "wb") as handle:
        pickle.dump(payload, handle)
    os.rename(temp_path, final_path)

    return {"document_id": document_id, "skipped": False, "node_count": len(nodes)}


async def main():
    STORAGE_ROOT.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH) as handle:
        manifest = json.load(handle)
    manifest = loop_template.apply_throttle(manifest)

    for entry in manifest:
        document_id = entry["document_id"]
        result = await build_index_for_document(document_id)
        status = "skipped (cached)" if result["skipped"] else f"indexed ({result['node_count']} nodes)"
        print(f"{document_id}: {status}")


if __name__ == "__main__":
    asyncio.run(main())
