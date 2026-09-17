from dataset_generation.section_grouper import group_sections


def _node(node_id, document_id, header, content="text", token_count=10):
    return {
        "node_id": node_id,
        "document_id": document_id,
        "parent_item_header": header,
        "node_type": "text",
        "source_page_num": 1,
        "content": content,
        "token_count": token_count,
    }


def test_groups_nodes_by_document_and_header():
    nodes = [
        _node("n1", "DOC_A", "Item 1A. Risk Factors", content="risk one", token_count=5),
        _node("n2", "DOC_A", "Item 1A. Risk Factors", content="risk two", token_count=7),
        _node("n3", "DOC_A", "Item 7. MD&A", content="md and a", token_count=9),
    ]
    sections = group_sections(nodes)
    assert len(sections) == 2
    risk_section = next(section for section in sections if "Risk Factors" in section["section_header"])
    assert risk_section["node_ids"] == ["n1", "n2"]
    assert risk_section["content"] == "risk one\n\nrisk two"
    assert risk_section["token_count"] == 12


def test_normalizes_header_case_into_one_section():
    nodes = [
        _node("n1", "DOC_A", "ITEM 1A. RISK FACTORS"),
        _node("n2", "DOC_A", "Item 1A. Risk Factors"),
    ]
    sections = group_sections(nodes)
    assert len(sections) == 1
    assert sections[0]["node_ids"] == ["n1", "n2"]


def test_excludes_null_and_empty_headers():
    nodes = [
        _node("n1", "DOC_A", None),
        _node("n2", "DOC_A", ""),
        _node("n3", "DOC_A", "   "),
        _node("n4", "DOC_A", "Item 8. Financial Statements"),
    ]
    sections = group_sections(nodes)
    assert len(sections) == 1
    assert sections[0]["node_ids"] == ["n4"]


def test_keeps_documents_separate():
    nodes = [
        _node("n1", "DOC_A", "Item 1A. Risk Factors"),
        _node("n2", "DOC_B", "Item 1A. Risk Factors"),
    ]
    sections = group_sections(nodes)
    assert len(sections) == 2
    assert {section["document_id"] for section in sections} == {"DOC_A", "DOC_B"}
