"""Writes golden_queries_to_label.md so a researcher can hand-write the
'why this answer is good' note and 0-100 score for each of the 20 golden
queries.

write_gq_labels.py is the authority for the current calibration set. The
markdown template this module writes has no example_output field, so
re-importing a stale golden_queries_to_label.md would overwrite
human_score, human_reasoning and is_good with old values while leaving
example_output exactly as write_gq_labels.py left it, silently mismatching
the two halves of the calibration."""
import asyncio

import database_manager as dbm


def render_label_markdown(golden_queries: list[dict]) -> str:
    parts = ["# Golden Queries -- Human Labeling\n"]
    for golden_query in golden_queries:
        parts.append(
            f"## {golden_query['query_id']}\n\n"
            f"**Quadrant:** {golden_query['quadrant']}\n"
            f"**Document:** {golden_query['document_id']}\n"
            f"**Query:** {golden_query['query_text']}\n"
            f"**Ground truth:** {golden_query['ground_truth_answer']}\n\n"
            f"**Score (0-100):** \n"
            f"**Good Example (yes/no):** \n"
            f"**Why this answer is good:** \n\n"
            f"---\n"
        )
    return "\n".join(parts)


async def main(db_path: str = "benchmark.db", out_path: str = "golden_queries_to_label.md") -> None:
    golden_queries = await dbm.get_golden_queries(db_path)
    markdown = render_label_markdown(golden_queries)
    with open(out_path, "w") as handle:
        handle.write(markdown)
    print(f"Wrote {len(golden_queries)} golden queries to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
