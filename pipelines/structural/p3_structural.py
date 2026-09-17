"""P3: structural summary-tree retrieval.

Loads the per-filing TreeIndex built by build_summary_index.py and
traverses it root-to-leaf by embedding similarity against the LLM-written
summary nodes. No LLM call happens in this class; only the tree build
step is allowed to call one.
"""
from pathlib import Path

from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.indices.tree import TreeIndex
from llama_index.core.llms import MockLLM
from llama_index.core.schema import MetadataMode, NodeWithScore
from llama_index.embeddings.fastembed import FastEmbedEmbedding

import pipelines.structural.build_summary_index as bsi
from pipelines.base import Retriever
from pipelines.vector.build_vector_index import EMBED_MODEL_NAME
from model_cache import model_cache_dir

_RETRIEVER_MODE = "select_leaf_embedding"


class P3StructuralRetriever(Retriever):
    def __init__(self, storage_root: Path | None = None):
        # storage_root defaults to bsi.STORAGE_ROOT resolved here, at call
        # time, rather than bound as a def-time default, so tests that
        # monkeypatch bsi.STORAGE_ROOT are honoured even when the caller
        # omits storage_root.
        if storage_root is None:
            storage_root = bsi.STORAGE_ROOT
        self._storage_root = storage_root
        self._embed_model = FastEmbedEmbedding(model_name=EMBED_MODEL_NAME, cache_dir=model_cache_dir())
        self._index_cache: dict[str, TreeIndex] = {}

    def _load_index(self, document_id: str) -> TreeIndex:
        if document_id in self._index_cache:
            return self._index_cache[document_id]

        persist_dir = self._storage_root / document_id
        if not persist_dir.exists():
            raise FileNotFoundError(
                f"No summary index found for document_id={document_id!r} at "
                f"{persist_dir}. Run build_summary_index.py for this filing "
                "before querying it."
            )
        storage_context = StorageContext.from_defaults(persist_dir=str(persist_dir))
        # MockLLM is bound deliberately: TreeIndex hands its own llm to every
        # retriever it builds, and an unset llm silently resolves to whatever
        # Settings/env defaults to, so a stray query-time LLM call could reach
        # a real paid endpoint. The mock keeps that impossible while the
        # embedding traversal path below uses no LLM at all.
        index = load_index_from_storage(storage_context, llm=MockLLM())
        self._index_cache[document_id] = index
        return index

    def retrieve(self, query_text: str, document_id: str, k: int) -> list[NodeWithScore]:
        index = self._load_index(document_id)
        # child_branch_factor is P3's k lever: the traversal keeps the k most
        # similar nodes at each level, so the leaf level yields at most k.
        tree_retriever = index.as_retriever(
            retriever_mode=_RETRIEVER_MODE,
            embed_model=self._embed_model,
            child_branch_factor=k,
        )
        candidates = tree_retriever.retrieve(query_text)

        # Each index is built per filing, so a cross-filing leaf should be
        # impossible; the check enforces the Retriever contract regardless.
        nodes = [
            candidate.node
            for candidate in candidates
            if candidate.node.metadata.get("document_id") == document_id
        ]

        # Tree traversal itself returns unscored nodes, so scores are recomputed
        # here against the same local embedding model, giving P1-comparable
        # similarity values and a relevance ordering over the selected leaves.
        query_embedding = self._embed_model.get_query_embedding(query_text)
        results = []
        for node in nodes:
            node_embedding = node.embedding or self._embed_model.get_text_embedding(
                node.get_content(metadata_mode=MetadataMode.EMBED)
            )
            score = self._embed_model.similarity(query_embedding, node_embedding)
            results.append(NodeWithScore(node=node, score=float(score)))

        # Stable sort, so equally scored leaves keep their traversal order and
        # repeated calls return an identical ranking.
        results.sort(key=lambda result: -result.score)
        return results[:k]
