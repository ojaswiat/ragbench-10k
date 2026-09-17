"""Applies the Judge's calibration labels to golden_queries.

The 20 exemplars are the only thing that teaches the Judge what each score
on the 0-100 scale means, so the set deliberately spans the range and
carries a deliberately imperfect answer for every quadrant. Guardrails 4a
shows the Judge only the 5 exemplars matching the target row's quadrant,
which is why each quadrant needs its own low anchor rather than the set
merely balancing overall.

Self-contained: every LABELS entry, good or bad, carries its own
example_output string. Re-running this module against a fresh database
reproduces the full calibration set from scratch; it never depends on a
candidate answer already sitting in golden_queries from an earlier step.

Idempotent: re-running it rewrites the same values.
"""
import asyncio

import database_manager as dbm

LABELS: list[dict] = [
    # ------------------------------------------------------------------
    # Q1_Direct_Text
    # Injected failure: right topic, wrong specific value lifted from a
    # neighbouring sentence.
    # ------------------------------------------------------------------
    {
        "query_id": "QT1_GQ_001",
        "example_output": "December 7, 2023",
        "is_good": True,
        "human_score": 100,
        "human_reasoning": (
            "States the exact date the filing gives, with no hedging and no added "
            "detail that the source does not support. For a direct-text lookup "
            "this is the whole task, so it earns full marks."
        ),
    },
    {
        "query_id": "QT1_GQ_002",
        "example_output": "up to 115,500 shares",
        "is_good": True,
        "human_score": 90,
        "human_reasoning": (
            "Reproduces the share figure digit for digit and keeps the 'up to' "
            "wording, which matters because the arrangement sets a ceiling rather "
            "than a fixed quantity. It stops just short of full marks because it "
            "does not tie the figure back to the named trading arrangement, so a "
            "reader holding several plans cannot confirm the row from the answer "
            "alone."
        ),
    },
    {
        "query_id": "QT1_GQ_003",
        "example_output": (
            "The information called for by Item 14 is incorporated herein by "
            "reference to the material under the captions Item 1. Election of "
            "Directors - Related person transactions and Director independence "
            "in the Proxy Statement."
        ),
        "is_good": False,
        "human_score": 45,
        "human_reasoning": (
            "Lands in the right part of the filing and answers the 'where' half "
            "correctly, naming the Proxy Statement and using the filing's own "
            "incorporation-by-reference phrasing. The caption, though, is lifted "
            "from the sentence immediately above, which covers Item 13, so the "
            "answer points a reader at the related-person-transactions material "
            "instead of the auditor ratification material. Half right in form, "
            "wrong on the detail that distinguishes Item 14 from its neighbour, "
            "so it sits in the middle of the scale."
        ),
    },
    {
        "query_id": "QT1_GQ_004",
        "example_output": "Equity Compensation Plan Information",
        "is_good": False,
        "human_score": 15,
        "human_reasoning": (
            "Quotes a caption that genuinely appears in the correct sentence, so "
            "the retrieval worked, but returns the last of the three listed "
            "captions rather than the first. Ordering is the entire content of "
            "the question, so once that is wrong there is nothing left to credit; "
            "the answer is also stated flatly, giving a reader no cue that a list "
            "position was ever in play."
        ),
    },
    {
        "query_id": "QT1_GQ_005",
        "example_output": "within 120 days after the end of the fiscal year to which this report relates.",
        "is_good": True,
        "human_score": 93,
        "human_reasoning": (
            "Gives the exact 120-day window and carries the qualifying clause that "
            "ties the deadline to the fiscal year this report covers, so nothing "
            "material is dropped. It is a shade below top marks only on "
            "presentation: the sentence is lifted whole rather than framed as an "
            "answer to the question asked."
        ),
    },
    # ------------------------------------------------------------------
    # Q2_Implicit_Text
    # Injected failure: only one of the two facts the question needs
    # joining, stated confidently.
    # ------------------------------------------------------------------
    {
        "query_id": "QT2_GQ_001",
        "example_output": "Apple Inc., 2023 Form 10-K, page 19",
        "is_good": True,
        "human_score": 97,
        "human_reasoning": (
            "Joins all three header fields the question asks for, in the order it "
            "asks for them, and each matches the source exactly. The join is the "
            "work here and it is done cleanly, with no invented context around it."
        ),
    },
    {
        "query_id": "QT2_GQ_002",
        "example_output": (
            "The Company states that the Compensation Committee report is "
            "incorporated by reference but is deemed furnished, not filed, and is "
            "not incorporated by reference into any filing under the Securities "
            "Act of 1933 or the Securities Exchange Act of 1934, except to the "
            "extent the Company specifically incorporates it by reference."
        ),
        "is_good": True,
        "human_score": 88,
        "human_reasoning": (
            "Carries both halves of the question: the report is incorporated by "
            "reference, and it is deemed furnished rather than filed, with the "
            "carve-out for material the Company specifically incorporates. The "
            "furnished-not-filed distinction is the legally load-bearing point and "
            "it survives intact. Marked down slightly because the phrasing tracks "
            "the filing so closely that it restates rather than resolves the "
            "relationship between the two statutes."
        ),
    },
    {
        "query_id": "QT2_GQ_003",
        "example_output": (
            "Management concluded that Tesla's disclosure controls and procedures "
            "and its internal control over financial reporting were both effective "
            "as of December 31, 2023."
        ),
        "is_good": False,
        "human_score": 50,
        "human_reasoning": (
            "Everything it says is true and precisely dated, and it covers the "
            "effectiveness conclusion for both control systems. What it never "
            "reaches is the second half of the question: the inherent limitations "
            "management acknowledged, that controls may fail to prevent or detect "
            "misstatements and can degrade as conditions change. Stating one of "
            "two required facts confidently and stopping is exactly half the task, "
            "and the omission is invisible to a reader, so it scores mid-scale "
            "rather than higher."
        ),
    },
    {
        "query_id": "QT2_GQ_004",
        "example_output": (
            "JPMorgan Chase used the COSO framework to assess its internal control "
            "over financial reporting as of December 31, 2023."
        ),
        "is_good": False,
        "human_score": 15,
        "human_reasoning": (
            "Answers only the framework half, and even there drops the 2013 "
            "edition that the filing names, which is the detail that fixes which "
            "COSO framework was applied. The conclusion the Chairman, CEO and CFO "
            "reached about disclosure controls, the half a reader is most likely "
            "to be relying on, is absent altogether and is not flagged as absent. "
            "One imprecise fact out of two, delivered as though it were the whole "
            "answer, belongs near the bottom of the scale."
        ),
    },
    {
        "query_id": "QT2_GQ_005",
        "example_output": (
            "The information required by Item 13 is incorporated by reference to "
            "the Proxy Statement, under the caption 'Director independence'."
        ),
        "is_good": False,
        "human_score": 25,
        "human_reasoning": (
            "Gets the destination right and quotes one caption accurately, so it "
            "is not fabricated. But the question asks which captions are cited, "
            "plural, and the answer silently drops 'Item 1. Election of Directors "
            "- Related person transactions', the caption that actually covers the "
            "related-party disclosure Item 13 exists for. Presenting a partial "
            "citation list as complete sends a reader to the wrong section and "
            "leaves them with no reason to look further."
        ),
    },
    # ------------------------------------------------------------------
    # Q3_Direct_Table
    # Injected failure: correct row, adjacent column's figure.
    # ------------------------------------------------------------------
    {
        "query_id": "QT3_GQ_001",
        "example_output": "40",
        "is_good": True,
        "human_score": 97,
        "human_reasoning": (
            "Returns the single cell the question asks for, with none of the "
            "surrounding index rows dragged in alongside it. Picking one page "
            "number out of a long contents table without drifting to a neighbouring "
            "item is the whole of a direct table lookup, and it is done exactly."
        ),
    },
    {
        "query_id": "QT3_GQ_002",
        "example_output": "1,603,278",
        "is_good": True,
        "human_score": 90,
        "human_reasoning": (
            "Reproduces the total share count exactly, separators included, and "
            "correctly takes the fourth-quarter total rather than one of the "
            "monthly sub-rows stacked above it, which is the trap in this table. "
            "Slightly short of full marks because it gives the bare figure without "
            "naming the period, so a reader cannot confirm which row it came from."
        ),
    },
    {
        "query_id": "QT3_GQ_003",
        "example_output": (
            "Currency reduced worldwide sales by 0.9% in 2024, according to the "
            "Sales increase/(decrease) due to table."
        ),
        "is_good": False,
        "human_score": 45,
        "human_reasoning": (
            "Finds the right table, the right row and the right direction: currency "
            "was a drag on sales, not a contributor, and the answer says so. The "
            "figure, however, is read one column across, from the 2023 comparative "
            "instead of the 2024 column the question names. Because the two years "
            "are close in magnitude the qualitative conclusion a reader draws is "
            "still broadly correct, which is why this sits mid-scale rather than at "
            "the bottom, but the number asked for is wrong."
        ),
    },
    {
        "query_id": "QT3_GQ_004",
        "example_output": "$ 139",
        "is_good": False,
        "human_score": 20,
        "human_reasoning": (
            "Correct row, adjacent column: it takes the term debt line but reads "
            "the 2024 comparative rather than the fiscal 2025 figure the question "
            "explicitly pins. Nothing beyond the row selection is right, and the "
            "answer is a bare number with no year, instrument or unit attached, so "
            "a reader has no cue that a column slipped. The ten-million gap between "
            "the two columns is small enough to pass unchallenged, which makes the "
            "error more costly rather than less."
        ),
    },
    {
        "query_id": "QT3_GQ_005",
        "example_output": "$ (1,343)",
        "is_good": True,
        "human_score": 93,
        "human_reasoning": (
            "Takes the correct sensitivity row and, critically, keeps the "
            "parentheses that mark the figure as a decline in fair value. Sign "
            "convention is the detail most often lost when a financial table is "
            "flattened into text, and preserving it is what separates a usable "
            "answer from a misleading one here. Held just below full marks because "
            "it does not restate the millions unit carried in the table header."
        ),
    },
    # ------------------------------------------------------------------
    # Q4_Implicit_Table
    # Injected failure: arithmetic performed on the right rows but the sign
    # or scale is wrong.
    # ------------------------------------------------------------------
    {
        "query_id": "QT4_GQ_001",
        "example_output": "$(1,868)",
        "is_good": True,
        "human_score": 98,
        "human_reasoning": (
            "Selects the three sensitivity rows the question specifies, sums them "
            "correctly, and reports the total in the parenthesised convention the "
            "table uses, so the result still reads as a loss. Both halves of the "
            "task, choosing the right rows and preserving the sign through the "
            "arithmetic, are done, which is what a top score means for an implicit "
            "table question."
        ),
    },
    {
        "query_id": "QT4_GQ_002",
        "example_output": "14",
        "is_good": True,
        "human_score": 93,
        "human_reasoning": (
            "Counts only the signatory rows and excludes the header and title rows "
            "that surround them, which is the whole difficulty in a signature block "
            "flattened out of its original layout. The count matches the filing "
            "exactly. Kept just below full marks because the bare number shows none "
            "of the boundary it used, so the count cannot be audited from the answer."
        ),
    },
    {
        "query_id": "QT4_GQ_003",
        "example_output": "160%",
        "is_good": False,
        "human_score": 50,
        "human_reasoning": (
            "Reads the right row and the right pair of endpoints, September 2018 "
            "and September 2023, and does not confuse the S&P 500 line with the "
            "Apple or technology-index lines beside it. The error is one of scale: "
            "it reports the indexed value itself as the percentage increase, never "
            "subtracting the $100 base the table starts every series from. The "
            "retrieval and row selection are sound and the final step is not, so it "
            "sits mid-scale."
        ),
    },
    {
        "query_id": "QT4_GQ_004",
        "example_output": "8",
        "is_good": False,
        "human_score": 18,
        "human_reasoning": (
            "Counts across the correct table but treats the asterisk footnote "
            "printed beneath it as an eighth facility, so the tally overruns by "
            "one. For a counting question the count is the entire answer, and this "
            "one is stated as a plain integer with nothing a reader could check it "
            "against. The failure is a chunk-boundary artefact rather than a "
            "reading error, which is precisely the kind of mistake the benchmark "
            "should score harshly."
        ),
    },
    {
        "query_id": "QT4_GQ_005",
        "example_output": "71.0%",
        "is_good": False,
        "human_score": 22,
        "human_reasoning": (
            "Locates the correct row and the correct leased-international cell, "
            "then divides by the wrong base, using the international subtotal "
            "instead of the total facilities figure the question asks about. The "
            "result is a plausible-looking percentage that answers a different "
            "question, what share of international facilities are leased, and "
            "overstates the true proportion by a factor of about two and a half. "
            "A denominator error of this kind is invisible in the output, so it "
            "earns a low score despite the underlying retrieval being correct."
        ),
    },
]


async def main(db_path: str = "benchmark.db") -> int:
    for label in LABELS:
        if label["example_output"] is not None:
            await dbm.update_golden_query_example_output(
                db_path, label["query_id"], label["example_output"]
            )
        await dbm.update_golden_query_labels(
            db_path,
            label["query_id"],
            label["human_score"],
            label["human_reasoning"],
            label["is_good"],
        )
    print(f"Applied {len(LABELS)} calibration labels")
    return len(LABELS)


if __name__ == "__main__":
    asyncio.run(main())
