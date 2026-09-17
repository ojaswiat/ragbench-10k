import pipelines.structural.node_convert as node_convert


def test_nodes_to_llama_nodes_preserves_id_and_text():
    nodes = [
        {"node_id": "AAPL_2025_n0001", "document_id": "AAPL_2025",
         "parent_item_header": "Item 1A. Risk Factors", "node_type": "text",
         "source_page_num": None, "content": "hello world", "token_count": 2},
    ]
    result = node_convert.nodes_to_llama_nodes(nodes)
    assert len(result) == 1
    assert result[0].id_ == "AAPL_2025_n0001"
    assert result[0].text == "hello world"


def test_nodes_to_llama_nodes_preserves_metadata():
    nodes = [
        {"node_id": "AAPL_2025_n0002", "document_id": "AAPL_2025",
         "parent_item_header": None, "node_type": "table",
         "source_page_num": None, "content": "| a | b |", "token_count": 3},
    ]
    result = node_convert.nodes_to_llama_nodes(nodes)
    assert result[0].metadata["document_id"] == "AAPL_2025"
    assert result[0].metadata["node_type"] == "table"
    assert result[0].metadata["parent_item_header"] is None


def test_nodes_to_llama_nodes_empty_list():
    assert node_convert.nodes_to_llama_nodes([]) == []
