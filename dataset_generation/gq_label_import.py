"""Reads the filled-in golden_queries_to_label.md back into golden_queries.

write_gq_labels.py is the authority for the current calibration set. The
markdown template this module reads has no example_output field, so it never
writes that column. Importing a stale golden_queries_to_label.md (one
predating write_gq_labels.py's calibration) would overwrite human_score,
human_reasoning and is_good with old values while leaving example_output
exactly as write_gq_labels.py left it, silently mismatching the two halves
of the calibration."""
import asyncio
import re

from core import database_manager as dbm

_ENTRY_RE = re.compile(
    r"^## (?P<query_id>\S+)\n\n"
    r".*?"
    r"\*\*Score \(0-100\):\*\*\s*(?P<score>\S+)?\s*\n"
    r"(?:\*\*Good Example \(yes/no\):\*\*\s*(?P<is_good>\S+)?\s*\n)?"
    r"\*\*Why this answer is good:\*\*\s*(?P<reasoning>.*?)\s*\n",
    re.DOTALL | re.MULTILINE,
)


def parse_label_markdown(markdown_text: str) -> list[dict]:
    results = []
    for match in _ENTRY_RE.finditer(markdown_text):
        score_raw = match.group("score")
        reasoning = (match.group("reasoning") or "").strip()
        if not score_raw or not reasoning:
            continue
        query_id = match.group("query_id")
        try:
            score = int(score_raw)
        except ValueError:
            raise ValueError(f"{query_id}: score {score_raw!r} is not an integer")

        is_good_raw = (match.group("is_good") or "").strip().lower() or None
        if is_good_raw is None:
            is_good = None
        elif is_good_raw == "yes":
            is_good = True
        elif is_good_raw == "no":
            is_good = False
        else:
            raise ValueError(f"{query_id}: is_good {is_good_raw!r} is not 'yes' or 'no'")

        results.append({
            "query_id": query_id,
            "human_score": score,
            "human_reasoning": reasoning,
            "is_good": is_good,
        })
    return results


async def main(db_path: str = "benchmark.db", in_path: str = "golden_queries_to_label.md") -> None:
    with open(in_path) as handle:
        markdown_text = handle.read()

    labels = parse_label_markdown(markdown_text)
    for index, label in enumerate(labels):
        try:
            await dbm.update_golden_query_labels(
                db_path, label["query_id"], label["human_score"], label["human_reasoning"],
                label.get("is_good"),
            )
        except Exception:
            print(f"Imported {index} of {len(labels)} labels before failing on {label['query_id']!r}")
            raise
    print(f"Imported {len(labels)} human labels from {in_path}")


if __name__ == "__main__":
    asyncio.run(main())
