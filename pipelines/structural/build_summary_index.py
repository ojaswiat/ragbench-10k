"""Builds and persists one hierarchical TreeIndex per filing.

This is the only index build permitted to call an LLM; it runs once per
filing and is cached, never rebuilt per query. LLM calls route through
LLMFactory for multi-provider routing and rate limiting.
"""
import asyncio
import json
import os
import shutil
import time
from pathlib import Path

from llama_index.core import TreeIndex
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler

import database_manager as dbm
import loop_template
from pipelines.structural.node_convert import nodes_to_llama_nodes
from llm_client.llm_factory import LLMFactory

STORAGE_ROOT = Path("storage/summary_index")
DB_PATH = "benchmark.db"
COST_LOG_PATH = Path("logs/index_build_costs.json")
MANIFEST_PATH = "data/filings_manifest.json"


def _final_dir(document_id: str) -> Path:
    return STORAGE_ROOT / document_id


def _temp_dir(document_id: str) -> Path:
    return STORAGE_ROOT / f"{document_id}.tmp"


def is_built(document_id: str) -> bool:
    return _final_dir(document_id).exists()


async def build_index_for_document(document_id: str) -> dict:
    if is_built(document_id):
        return {"document_id": document_id, "skipped": True}

    nodes = await dbm.get_nodes_by_document(DB_PATH, document_id)
    if not nodes:
        raise ValueError(
            f"No nodes found for document_id={document_id!r} -- this filing "
            "has not been ingested yet (Phase 2). Refusing to build/persist "
            "an empty TreeIndex, which would create a permanent false-positive "
            "cache hit in storage/summary_index/."
        )
    print(f"Started: {document_id} ({len(nodes)} nodes)", flush=True)
    llama_nodes = nodes_to_llama_nodes(nodes)

    token_counter = TokenCountingHandler()
    callback_manager = CallbackManager([token_counter])

    summary_agent = LLMFactory.get_client_for_stage(
        "p3_index_build", callback_manager=callback_manager
    )
    start = time.monotonic()
    index = TreeIndex(
        nodes=llama_nodes,
        llm=summary_agent,
        callback_manager=callback_manager,
        build_tree=True,
    )
    wall_clock_sec = time.monotonic() - start

    temp_dir = _temp_dir(document_id)
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    index.storage_context.persist(persist_dir=str(temp_dir))
    os.rename(temp_dir, _final_dir(document_id))
    print(f"Finished: {document_id} ({wall_clock_sec:.1f}s)", flush=True)

    return {
        "document_id": document_id,
        "skipped": False,
        "wall_clock_sec": wall_clock_sec,
        "input_tokens": token_counter.prompt_llm_token_count,
        "output_tokens": token_counter.completion_llm_token_count,
    }


def append_cost_log(cost_row: dict, log_path: Path = COST_LOG_PATH) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    rows = json.loads(log_path.read_text()) if log_path.exists() else []
    rows.append(cost_row)
    log_path.write_text(json.dumps(rows, indent=2))


def confirm_build(document_id: str, node_count: int) -> bool:
    print(f"The document {document_id} contains {node_count} nodes")
    while True:
        answer = input("Y: Build it\nN: Skip it\n> ").strip().lower()
        if answer == "y":
            return True
        if answer == "n":
            return False
        print("Please enter Y or N.")


async def main():
    STORAGE_ROOT.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH) as handle:
        manifest = json.load(handle)
    manifest = loop_template.apply_throttle(manifest)

    for entry in manifest:
        document_id = entry["document_id"]

        if is_built(document_id):
            cost_row = await build_index_for_document(document_id)
            append_cost_log(cost_row)
            print(f"{document_id}: skipped (cached)")
            continue

        nodes = await dbm.get_nodes_by_document(DB_PATH, document_id)
        if nodes and not confirm_build(document_id, len(nodes)):
            print(f"{document_id}: skipped (declined)")
            continue

        cost_row = await build_index_for_document(document_id)
        append_cost_log(cost_row)
        status = "skipped (cached)" if cost_row.get("skipped") else "built"
        print(f"{document_id}: {status}")


if __name__ == "__main__":
    asyncio.run(main())
