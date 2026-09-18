"""Builds the shared P1 vector index: one Chroma collection,
metadata-tagged by document_id, covering every filing's nodes.

Fully local: embeddings via fastembed (BAAI/bge-small-en-v1.5), no LLM
call. Resumable at document granularity.
"""
import asyncio
import json
from pathlib import Path

import chromadb
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.schema import NodeRelationship, RelatedNodeInfo
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from core import database_manager as dbm
from core import loop_template
from pipelines.structural.node_convert import nodes_to_llama_nodes
from core.model_cache import model_cache_dir

STORAGE_ROOT = Path("storage/chroma")
COLLECTION_NAME = "p1_vector"
DB_PATH = "benchmark.db"
MANIFEST_PATH = "data/filings_manifest.json"
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# nodes_to_llama_nodes (shared with P3) puts these keys in
# TextNode.metadata. P1 must embed only the raw content: letting
# node_type or parent_item_header leak into the embedded text would give
# the vector pipeline structural signal that P2 and P3 don't get,
# breaking comparability between the three pipelines. Excluded from both
# embed and LLM metadata modes for uniform handling, even though no LLM
# call happens in this pipeline.
_STRUCTURAL_METADATA_KEYS = [
    "parent_item_header",
    "node_type",
    "source_page_num",
    "document_id",
]


def get_collection(storage_root: Path | None = None):
    # storage_root defaults to the module-level STORAGE_ROOT, resolved at
    # call time rather than bound as a default argument, so tests can
    # monkeypatch STORAGE_ROOT and have every call pick up the override.
    if storage_root is None:
        storage_root = STORAGE_ROOT
    client = chromadb.PersistentClient(path=str(storage_root))
    return client.get_or_create_collection(COLLECTION_NAME)


def is_document_indexed(document_id: str, storage_root: Path | None = None) -> bool:
    collection = get_collection(storage_root)
    existing = collection.get(where={"document_id": document_id}, limit=1)
    return len(existing["ids"]) > 0


async def build_index_for_document(
    document_id: str,
    db_path: str = DB_PATH,
    storage_root: Path | None = None,
) -> dict:
    if is_document_indexed(document_id, storage_root=storage_root):
        return {"document_id": document_id, "skipped": True, "node_count": 0}

    nodes = await dbm.get_nodes_by_document(db_path, document_id)
    if not nodes:
        # An empty node list would still be a "successful" no-op build,
        # leaving is_document_indexed() permanently False and masking a
        # real ingestion gap behind what looks like a not-yet-run build.
        # Raising here forces the gap to surface immediately instead.
        raise ValueError(
            f"No nodes found for document_id={document_id!r}. Run Phase 2 "
            "ingestion for this filing before building the vector index."
        )

    llama_nodes = nodes_to_llama_nodes(nodes)
    # Strip structural metadata before it can reach the embedding text,
    # see _STRUCTURAL_METADATA_KEYS above. Only node.text (raw content) is
    # embedded after this.
    for node in llama_nodes:
        node.excluded_embed_metadata_keys = list(_STRUCTURAL_METADATA_KEYS)
        node.excluded_llm_metadata_keys = list(_STRUCTURAL_METADATA_KEYS)
    # ChromaVectorStore writes node.ref_doc_id into its own reserved
    # top-level "document_id" metadata field (llama_index.core.vector_stores
    # .utils.node_to_metadata_dict), independent of our custom
    # metadata["document_id"] set by nodes_to_llama_nodes. Without a source
    # relationship, ref_doc_id is None and that reserved field overwrites
    # ours with the literal string "None", breaking is_document_indexed's
    # where-filter. Set the SOURCE relationship so both agree.
    for node in llama_nodes:
        node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(
            node_id=document_id
        )

    collection = get_collection(storage_root)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    embed_model = FastEmbedEmbedding(model_name=EMBED_MODEL_NAME, cache_dir=model_cache_dir())

    VectorStoreIndex(
        nodes=llama_nodes,
        storage_context=storage_context,
        embed_model=embed_model,
    )

    # Chroma's collection.add() silently keeps the OLD row and drops the
    # new one on a duplicate id, instead of erroring or overwriting. If any
    # of this document's node ids already existed in the collection under a
    # different (or missing) document_id, for example leftover rows from a
    # prior run that used different node ids for the same slot, the write
    # above would have completed without indexing all of this document's
    # nodes, with no exception raised anywhere. Verifying the row count
    # here turns that silent gap into a hard failure at build time.
    indexed = collection.get(where={"document_id": document_id})
    if len(indexed["ids"]) != len(llama_nodes):
        raise RuntimeError(
            f"Indexing verification failed for document_id={document_id!r}: "
            f"expected {len(llama_nodes)} rows in the collection, found "
            f"{len(indexed['ids'])}. This usually means pre-existing rows "
            "under colliding node ids silently blocked the new write -- "
            "inspect the Chroma collection for stale/duplicate ids before "
            "retrying."
        )

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
