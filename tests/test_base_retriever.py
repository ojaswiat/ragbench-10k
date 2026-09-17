import pytest
from llama_index.core.schema import NodeWithScore

from pipelines.base import Retriever


def test_retriever_is_abstract_and_cannot_be_instantiated():
    with pytest.raises(TypeError):
        Retriever()


def test_concrete_subclass_must_implement_retrieve():
    class Incomplete(Retriever):
        pass

    with pytest.raises(TypeError):
        Incomplete()


def test_concrete_subclass_with_retrieve_can_be_instantiated_and_called():
    class Dummy(Retriever):
        def retrieve(self, query_text: str, document_id: str, k: int) -> list[NodeWithScore]:
            return []

    instance = Dummy()
    assert instance.retrieve("q", "AAPL_2023", 5) == []
