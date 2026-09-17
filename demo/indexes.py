"""Ensures the demo filing has all three retrieval indexes on disk.

P3's summary tree was built in Phase 3 and is reused as-is. P1 and P2 build
lazily per document, so the demo only ever pays for one filing rather than
the whole corpus. Every build here is local CPU work: no API key is read and
no free-tier quota is spent.
"""
import asyncio
import time

import pipelines.bm25.build_bm25_index as bm25_builder
import pipelines.structural.build_summary_index as tree_builder
import pipelines.vector.build_vector_index as vector_builder

from demo.config import DEMO_DOCUMENT


def _p3_present(document_id: str) -> bool:
    return (tree_builder.STORAGE_ROOT / document_id).exists()


async def ensure_indexes(document_id: str = DEMO_DOCUMENT, log=print) -> dict:
    """Builds whatever is missing and reports what each step cost."""
    report = {}

    log(f"P3 structural  : checking storage/summary_index/{document_id}")
    if _p3_present(document_id):
        log("                 already built in Phase 3, reused as-is")
        report["P3_structural"] = "reused"
    else:
        raise FileNotFoundError(
            f"No P3 summary tree for {document_id}. That index is LLM-built and "
            "cannot be regenerated offline; pick a filing that already has one."
        )

    log(f"P2 statistical : checking storage/bm25/{document_id}.pkl")
    if bm25_builder.is_document_indexed(document_id):
        log("                 already built, reused")
        report["P2_bm25"] = "reused"
    else:
        log("                 building BM25 corpus (local, no network)")
        bm25_builder.STORAGE_ROOT.mkdir(parents=True, exist_ok=True)
        start = time.perf_counter()
        result = await bm25_builder.build_index_for_document(document_id)
        elapsed = time.perf_counter() - start
        log(f"                 built {result.get('node_count', 0)} nodes in {elapsed:.1f}s")
        report["P2_bm25"] = f"built in {elapsed:.1f}s"

    log(f"P1 semantic    : checking storage/chroma/{document_id}")
    if vector_builder.is_document_indexed(document_id):
        log("                 already built, reused")
        report["P1_vector"] = "reused"
    else:
        log("                 embedding nodes with bge-small-en-v1.5 (local CPU)")
        log("                 first run downloads the model, this is the slow step")
        vector_builder.STORAGE_ROOT.mkdir(parents=True, exist_ok=True)
        start = time.perf_counter()
        result = await vector_builder.build_index_for_document(document_id)
        elapsed = time.perf_counter() - start
        log(f"                 embedded {result.get('node_count', 0)} nodes in {elapsed:.1f}s")
        report["P1_vector"] = f"built in {elapsed:.1f}s"

    return report


if __name__ == "__main__":
    asyncio.run(ensure_indexes())
