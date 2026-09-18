"""Cross-encoder reranker for P1, backed by fastembed's ONNX
TextCrossEncoder.

No official llama-index postprocessor wraps fastembed's reranker, so
this is a small custom BaseNodePostprocessor around BAAI/bge-reranker-base.
"""
from fastembed.rerank.cross_encoder import TextCrossEncoder
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import MetadataMode, NodeWithScore, QueryBundle
from core.model_cache import model_cache_dir


class FastEmbedReranker(BaseNodePostprocessor):
    model_name: str = "BAAI/bge-reranker-base"
    top_n: int | None = None

    _model: TextCrossEncoder = PrivateAttr()

    def __init__(self, model_name: str = "BAAI/bge-reranker-base", top_n: int | None = None, **kwargs):
        super().__init__(model_name=model_name, top_n=top_n, **kwargs)
        self._model = TextCrossEncoder(model_name=model_name, cache_dir=model_cache_dir())

    @classmethod
    def class_name(cls) -> str:
        return "FastEmbedReranker"

    def _postprocess_nodes(
        self,
        nodes: list[NodeWithScore],
        query_bundle: QueryBundle | None = None,
    ) -> list[NodeWithScore]:
        if not nodes or query_bundle is None:
            return nodes

        # get_content() already defaults to MetadataMode.NONE (raw content
        # only); stated explicitly here so the reranker's input can't
        # silently start including metadata if that default ever changes.
        texts = [node.node.get_content(metadata_mode=MetadataMode.NONE) for node in nodes]
        scores = list(self._model.rerank(query_bundle.query_str, texts))

        for node, score in zip(nodes, scores):
            node.score = float(score)

        ranked = sorted(nodes, key=lambda scored_node: scored_node.score, reverse=True)
        return ranked[: self.top_n] if self.top_n is not None else ranked
