"""Shared retriever interface implemented by all three pipelines.

retrieve() must filter to the given document_id internally and never
return a node from another filing. document_id is a property of the
query itself, not a metadata pre-filter on where the answer lives, so
requiring it here leaks no answer-location information.
"""
from abc import ABC, abstractmethod

from llama_index.core.schema import NodeWithScore


class Retriever(ABC):
    @abstractmethod
    def retrieve(self, query_text: str, document_id: str, k: int) -> list[NodeWithScore]:
        ...
