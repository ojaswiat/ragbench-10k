"""Randomised parsing validation audit: samples nodes for a human to check
that tables and headers survived the parse intact.

Surfaces the sample only; a human must judge the result.
"""
import random

import aiosqlite


async def sample_sections(db_path: str, sample_size: int = 20) -> list[dict]:
    async with aiosqlite.connect(db_path) as conn:
        conn.row_factory = aiosqlite.Row
        cursor = await conn.execute("SELECT * FROM nodes")
        rows = [dict(row) for row in await cursor.fetchall()]

    if len(rows) <= sample_size:
        return rows
    return random.sample(rows, sample_size)


def write_audit_report(samples: list[dict], out_path: str) -> None:
    lines = ["# Parsing Validation Audit", "", f"Sampled {len(samples)} node(s) for manual review.", ""]
    for sample in samples:
        lines.append(f"## {sample['node_id']} ({sample['node_type']})")
        lines.append(f"- Document: {sample['document_id']}")
        lines.append(f"- Section: {sample.get('parent_item_header') or '(none)'}")
        lines.append(f"- Token count: {sample['token_count']}")
        lines.append("")
        lines.append("```")
        lines.append(sample["content"])
        lines.append("```")
        lines.append("")
    with open(out_path, "w") as handle:
        handle.write("\n".join(lines))
