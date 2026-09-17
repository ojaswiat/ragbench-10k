from dataset_generation.search_tool import SEARCH_TOOL_SCHEMA, search_filing_nodes


def _node(node_id, content):
    return {"node_id": node_id, "content": content}


def test_ranks_by_keyword_overlap():
    nodes = [
        _node("n1", "Total revenue increased due to strong iPhone sales"),
        _node("n2", "The board of directors met quarterly"),
        _node("n3", "Total revenue and total operating expenses both rose"),
    ]
    results = search_filing_nodes(nodes, "total revenue", top_k=5)
    assert [result["node_id"] for result in results] == ["n3", "n1"]


def test_excludes_zero_overlap_nodes():
    nodes = [
        _node("n1", "unrelated content about employees"),
    ]
    results = search_filing_nodes(nodes, "total revenue", top_k=5)
    assert results == []


def test_respects_top_k():
    nodes = [_node(f"n{index}", "revenue revenue revenue") for index in range(10)]
    results = search_filing_nodes(nodes, "revenue", top_k=3)
    assert len(results) == 3


def test_ties_broken_by_node_id():
    nodes = [
        _node("n9", "revenue figures"),
        _node("n2", "revenue figures"),
    ]
    results = search_filing_nodes(nodes, "revenue", top_k=5)
    assert [result["node_id"] for result in results] == ["n2", "n9"]


def test_tool_schema_shape():
    assert SEARCH_TOOL_SCHEMA["type"] == "function"
    assert SEARCH_TOOL_SCHEMA["function"]["name"] == "search_filing"
    assert "query" in SEARCH_TOOL_SCHEMA["function"]["parameters"]["properties"]


def test_short_precise_node_outranks_long_wordy_node():
    # Scoring is length-normalised TF (hits/length), not raw hit count, so
    # a short node with fewer hits can outrank a longer one with more. Both
    # nodes stay >= _MIN_NODE_LENGTH_FOR_SCORING (25 tokens) so the floor
    # does not mask the comparison.
    # Node A: 26 tokens, 1 "revenue" hit -> score 1/26 ~= 0.0385.
    node_a = _node(
        "nA",
        "Total revenue increased twelve percent year over year driven "
        "mainly by strong demand across the core product lines in every "
        "major region this quarter.",
    )
    # Node B: 60 tokens, 5 "revenue" hits -> score 5/60 ~= 0.0833.
    node_b = _node(
        "nB",
        "The company reported quarterly results showing that revenue "
        "growth remained steady across all business segments during the "
        "period analysts noted that revenue trends were broadly consistent "
        "with prior guidance while management commentary on revenue "
        "drivers highlighted strength in cloud services offset by "
        "softness elsewhere and the filing also discussed revenue "
        "recognition policies in detail alongside revenue guidance for "
        "next fiscal.",
    )
    results = search_filing_nodes([node_a, node_b], "revenue", top_k=5)
    assert [result["node_id"] for result in results] == ["nB", "nA"]


def test_empty_content_node_skipped_without_zero_division():
    nodes = [
        _node("n1", ""),
        _node("n2", "revenue figures"),
    ]
    results = search_filing_nodes(nodes, "revenue", top_k=5)
    assert [result["node_id"] for result in results] == ["n2"]


def test_degenerate_tiny_node_does_not_outrank_real_node():
    # Without a length floor, a 1-token node with one hit scores
    # raw_count/len == 1/1 == 1.0 and beats a longer, substantive node
    # mentioning the term once. The floor
    # (score = raw_count / max(len(node_tokens), _MIN_NODE_LENGTH_FOR_SCORING))
    # corrects this.
    tiny_node = _node("tiny", "Revenue")  # 1 token, 1 hit -> pre-floor score 1.0
    real_node = _node(
        "real",
        "During the fiscal year the company generated substantial revenue "
        "from its core operating segments while continuing to invest in "
        "research and development, manufacturing capacity, and customer "
        "support infrastructure across all major geographic markets served "
        "by the business today.",
    )  # ~45 tokens, 1 "revenue" hit -> floored score 1/45 ~= 0.022 (vs. tiny's 1/25 = 0.04... )
    results = search_filing_nodes([tiny_node, real_node], "revenue", top_k=5)
    assert results[0]["node_id"] == "tiny"
