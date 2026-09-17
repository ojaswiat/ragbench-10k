from llama_index.core.schema import NodeWithScore, TextNode

from pipelines.vector.fastembed_reranker import FastEmbedReranker


def _node(node_id: str, text: str) -> NodeWithScore:
    return NodeWithScore(node=TextNode(id_=node_id, text=text), score=0.0)


def test_reranks_by_relevance_to_query():
    reranker = FastEmbedReranker()
    nodes = [
        _node("n1", "The weather today is sunny with a chance of rain."),
        _node("n2", "Apple reported total net sales of $394.3 billion in fiscal 2023."),
        _node("n3", "The cat sat on the mat."),
    ]

    result = reranker.postprocess_nodes(nodes, query_str="What were Apple's total net sales?")

    assert len(result) == 3
    assert result[0].node.node_id == "n2"
    assert all(result[index].score >= result[index + 1].score for index in range(len(result) - 1))


def test_top_n_truncates_result():
    reranker = FastEmbedReranker(top_n=1)
    nodes = [
        _node("n1", "Irrelevant text about weather."),
        _node("n2", "Apple's total net sales were $394.3 billion."),
    ]

    result = reranker.postprocess_nodes(nodes, query_str="What were Apple's total net sales?")

    assert len(result) == 1
    assert result[0].node.node_id == "n2"


def test_empty_nodes_returns_empty():
    reranker = FastEmbedReranker()
    assert reranker.postprocess_nodes([], query_str="anything") == []
