"""ONE live, throttled smoke test proving critique_query() speaks Groq's real
function-calling protocol, not just the mocked shape in test_async_critic.py.

Skipped by default. To run: set RUN_LIVE_GROQ_TESTS=1 in the environment and
resolve a real GROQ_API_KEY via config.py, then run pytest as usual. Without
both, the test is skipped and no Groq quota is spent.
"""
import os

import pytest
from llama_index.core.base.llms.types import MessageRole, ToolCallBlock

import llm_client.config as config
from dataset_generation.async_critic import critique_query

_RUN_LIVE = os.getenv("RUN_LIVE_GROQ_TESTS") == "1"

# Frozen snapshot of 16 real JNJ_2023 nodes (Item 9A-9C to the start of
# Item 10/11), including several with no lexical overlap with the question
# so search_filing_nodes must actually rank/filter, not return everything.
_JNJ_2023_ITEM_9_SLICE = [
    {"node_id": "JNJ_2023_n1161", "content": "Not applicable."},
    {"node_id": "JNJ_2023_n1162", "content": "# Disclosure controls and procedures."},
    {
        "node_id": "JNJ_2023_n1163",
        "content": (
            "At the end of the period covered by this Report, the Company evaluated the "
            "effectiveness of the design and operation of its disclosure controls and "
            "procedures. The Company’s disclosure controls and procedures are designed "
            "to ensure that information required to be disclosed by the Company in the "
            "reports that it files or submits under the Exchange Act is recorded, "
            "processed, summarized and reported within the time periods specified in the "
            "SEC’s rules and forms. Disclosure controls and procedures include, without "
            "limitation, controls and procedures designed to ensure that information "
            "required to be disclosed by the Company in the reports that it files or "
            "submits under the Exchange Act is accumulated and communicated to the "
            "Company’s management, including its principal executive and principal "
            "financial officers, or persons performing similar functions, as appropriate "
            "to allow timely decisions regarding required disclosure. Joaquin Duato, "
            "Chairman and Chief Executive Officer, and Joseph J. Wolk, Executive Vice "
            "President, Chief Financial Officer, reviewed and participated in this "
            "evaluation. Based on this evaluation, Messrs. Duato and Wolk concluded that, "
            "as of the end of the period covered by this Report, the Company’s "
            "disclosure controls and procedures were effective."
        ),
    },
    {"node_id": "JNJ_2023_n1164", "content": "# Reports on internal control over financial reporting."},
    {
        "node_id": "JNJ_2023_n1165",
        "content": (
            "The information called for by this item is incorporated herein by reference "
            "to Management’s report on internal control over financial reporting, and "
            "the attestation regarding internal controls over financial reporting included "
            "in the report of independent registered public accounting firm included in "
            "Item 8 of this Report."
        ),
    },
    {"node_id": "JNJ_2023_n1166", "content": "# Changes in internal control over financial reporting."},
    {
        "node_id": "JNJ_2023_n1167",
        "content": (
            "During the fiscal quarter ended December 31, 2023, there were no changes in "
            "the Company’s internal control over financial reporting identified in "
            "connection with the evaluation required under Rules 13a-15 and 15d-15 under "
            "the Exchange Act that have materially affected, or are reasonably likely to "
            "materially affect, the Company’s internal control over financial "
            "reporting. The Company continues to monitor and assess the effectiveness of "
            "the design and operation of its disclosure controls and procedures."
        ),
    },
    {
        "node_id": "JNJ_2023_n1168",
        "content": (
            "The Company is implementing a multi-year, enterprise-wide initiative to "
            "integrate, simplify and standardize processes and systems for the human "
            "resources, information technology, procurement, supply chain and finance "
            "functions. These are enhancements to support the growth of the Company’s "
            "financial shared service capabilities and standardize financial systems. "
            "This initiative is not in response to any identified deficiency or weakness "
            "in the Company’s internal control over financial reporting. In response "
            "to this initiative, the Company has and will continue to align and streamline "
            "the design and operation of its financial control environment."
        ),
    },
    {
        "node_id": "JNJ_2023_n1169",
        "content": (
            "Securities trading plans of Directors and Executive Officers. During the "
            "fiscal fourth quarter of 2023, none of our directors or officers (as defined "
            "in Rule 16a-1(f) of the Exchange Act) informed us of the adoption or "
            "termination of a “Rule 10b5-1 trading arrangement” or “non-Rule "
            "10b5-1 trading arrangement,” each as defined in Item 408 of Regulation S-K."
        ),
    },
    {"node_id": "JNJ_2023_n1170", "content": "Not applicable."},
    {"node_id": "JNJ_2023_n1171", "content": "112 Jhonson&#x26;Jhonson"},
    {"node_id": "JNJ_2023_n1172", "content": "# Part III"},
    {
        "node_id": "JNJ_2023_n1173",
        "content": (
            "The information called for by this item is incorporated herein by reference "
            "to the discussion of the Audit Committee under the caption Item 1. Election "
            "of Directors - Board committees; and the material under the captions Item 1. "
            "Election of Directors and, if applicable, Delinquent Section 16(a) reporting "
            "in the Proxy Statement; and the material under the caption “Executive "
            "Officers of the Registrant” in Part I of this Report."
        ),
    },
    {
        "node_id": "JNJ_2023_n1174",
        "content": (
            "The Company’s Code of Business Conduct, which covers all employees "
            "(including the Chief Executive Officer, Chief Financial Officer and "
            "Controller), meets the requirements of the SEC rules promulgated under "
            "Section 406 of the Sarbanes-Oxley Act of 2002. The Code of Business Conduct "
            "is available on the Company’s website at www.jnj.com/code-of-business-conduct, "
            "and copies are available to shareholders without charge upon written request "
            "to the Secretary at the Company’s principal executive offices. Any "
            "substantive amendment to the Code of Business Conduct or any waiver of the "
            "Code granted to the Chief Executive Officer, the Chief Financial Officer or "
            "the Controller will be posted on the Company’s website within five "
            "business days (and retained on the website for at least one year)."
        ),
    },
    {
        "node_id": "JNJ_2023_n1175",
        "content": (
            "In addition, the Company has adopted a Code of Business Conduct & Ethics for "
            "Members of the Board of Directors and Executive Officers. The Code of "
            "Business Conduct & Ethics for Members of the Board of Directors and "
            "Executive Officers is available on the Company’s website, and copies are "
            "available to shareholders without charge upon written request to the "
            "Secretary at the Company’s principal executive offices. Any substantive "
            "amendment to the Code or any waiver of the Code granted to any member of the "
            "Board of Directors or any executive officer will be posted on the "
            "Company’s website within five business days (and retained on the "
            "website for at least one year)."
        ),
    },
    {
        "node_id": "JNJ_2023_n1176",
        "content": (
            "The information called for by this item is incorporated herein by reference "
            "to the material under the captions Item 1. Election of Directors – "
            "Director compensation, and Item 2. Compensation Committee report, "
            "Compensation discussion and analysis and Executive compensation tables in "
            "the Proxy Statement."
        ),
    },
]

# Guards against accidental expansion of the frozen node slice into
# something that would make the real API call larger or costlier.
_MAX_ALLOWED_NODES = 20

_QUESTION = (
    "During Johnson & Johnson's fiscal fourth quarter of 2023, did any of the "
    "company's directors or officers adopt or terminate a Rule 10b5-1 trading "
    "arrangement?"
)


def _assert_verdict_shape(result: dict) -> None:
    """The verdict fields critique_query promises, present and non-empty."""
    assert isinstance(result, dict)
    assert "computed_answer" in result
    assert "cited_node_ids" in result
    assert isinstance(result["cited_node_ids"], list)
    assert isinstance(result["computed_answer"], str) and result["computed_answer"].strip()


def _assert_cited_the_only_relevant_node(result: dict) -> None:
    """The only node mentioning "Rule 10b5-1" / "trading arrangement" is n1169,
    so a correct search-then-answer round trip must cite it."""
    assert "JNJ_2023_n1169" in result["cited_node_ids"]


def _assert_history_proves_a_tool_call(result: dict) -> None:
    """Direct proof from the message history, rather than inferring from the
    cited node, that the Critic really invoked search_filing."""
    assert "messages" in result
    assert isinstance(result["messages"], list) and result["messages"]
    tool_call_blocks = [
        block
        for message in result["messages"]
        if message.role == MessageRole.ASSISTANT
        for block in message.blocks
        if isinstance(block, ToolCallBlock)
    ]
    assert tool_call_blocks, (
        "Expected at least one assistant message carrying a ToolCallBlock in "
        "the real Groq message history, proving the Critic actually invoked "
        "search_filing rather than answering directly."
    )


@pytest.mark.live
@pytest.mark.skipif(
    not _RUN_LIVE,
    reason="Live Groq smoke test; set RUN_LIVE_GROQ_TESTS=1 to run (spends real Groq quota).",
)
@pytest.mark.skipif(
    not config.GROQ_API_KEY,
    reason="No GROQ_API_KEY resolved via config.py; cannot make a real Groq call.",
)
@pytest.mark.asyncio
async def test_critique_query_live_round_trip_against_real_groq_api():
    """Proves the tool-call message shape critique_query() constructs is
    accepted and honoured by the live Groq API, not just by the mocks in
    test_async_critic.py.

    Deliberately one API call: the assertions are grouped into helpers rather
    than split into separate tests, so a live run spends the quota once.
    """
    assert len(_JNJ_2023_ITEM_9_SLICE) <= _MAX_ALLOWED_NODES

    result = await critique_query(_QUESTION, _JNJ_2023_ITEM_9_SLICE, return_messages=True)

    _assert_verdict_shape(result)
    _assert_cited_the_only_relevant_node(result)
    _assert_history_proves_a_tool_call(result)
