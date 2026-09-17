"""Terminal rendering for the demo. Presentation only, no logic."""
from demo.config import K, RULE, THIN

LABELS = {
    "P1_vector": "P1  semantic",
    "P2_bm25": "P2  statistical",
    "P3_structural": "P3  structural",
}


def banner(title: str, subtitle: str = "") -> None:
    print()
    print(RULE)
    print(f"  {title}")
    if subtitle:
        print(f"  {subtitle}")
    print(RULE)


def print_step(title: str) -> None:
    print()
    print(THIN)
    print(f"  {title}")
    print(THIN)


def short_node_label(node_id: str, document_id: str) -> str:
    """Strip the repeated document prefix so the ranked lists stay readable."""
    prefix = document_id + "_"
    return node_id[len(prefix):] if node_id.startswith(prefix) else node_id


def query_block(index: int, total: int, outcome: dict, document_id: str,
                pipelines: tuple[str, ...]) -> None:
    banner(f"QUERY {index} of {total}   ·   {outcome['query_id']}   ·   {outcome['quadrant']}")
    text = outcome["query_text"]
    print(f'{text}')
    print()
    ground_truth = ", ".join(short_node_label(node_id, document_id)
                             for node_id in outcome["ground_truth"])
    print(f"  Ground truth     {ground_truth}")
    print()
    print(f"  {'':<17}{'ranked top-' + str(K):<30}{'P@' + str(K):>7}{'R@' + str(K):>8}{'latency':>10}")
    for name in pipelines:
        row = outcome["pipelines"][name]
        cells = "".join(
            f"{short_node_label(nid, document_id) + ('*' if hit else ''):<10}"
            for nid, hit in zip(row["node_ids"], row["hits"]))
        print(f"  {LABELS[name]:<17}{cells:<30}"
              f"{row['precision']:>7.2f}{row['recall']:>8.2f}{row['latency_sec']:>9.2f}s")
    sets = [set(outcome["pipelines"][pipeline_name]["node_ids"])
            for pipeline_name in pipelines]
    union = set().union(*sets)
    shared = set.intersection(*sets)
    print()
    print(f"  * marks a node that is in the ground truth for this query")
    print(f"  {len(union)} distinct nodes returned across the three pipelines, "
          f"{len(shared)} returned by all three")


def summary_block(summary: dict, pipelines: tuple[str, ...], n_queries: int) -> None:
    banner(f"SUMMARY   ·   mean over {n_queries} queries at K={K}")
    print(f"  {'':<17}{'Precision@' + str(K):>14}{'Recall@' + str(K):>12}{'Mean latency':>16}")
    print(f"  {'':<17}{'-' * 14:>14}{'-' * 12:>12}{'-' * 16:>16}")
    for name in pipelines:
        if name not in summary:
            continue
        stats = summary[name]
        print(f"  {LABELS[name]:<17}{stats['precision']:>14.2f}{stats['recall']:>12.2f}"
              f"{stats['latency_sec']:>15.2f}s")


def how_to_read() -> None:
    banner("HOW TO READ THIS")
    print("  P1 semantic     embeds every node and reranks with a cross-encoder.")
    print("                  Fails when the answer shares no vocabulary with the")
    print("                  question. Pays for the rerank in latency.")
    print()
    print("  P2 statistical  ranks by BM25 term overlap, no model at all.")
    print("                  Strong on exact figures and named terms, blind to")
    print("                  paraphrase. Effectively free to run.")
    print()
    print("  P3 structural   walks a summary tree root to leaf by embedding")
    print("                  similarity. It commits to a branch near the root, so")
    print("                  a query whose wording matches a different section can")
    print("                  be routed away from the answer and never recover.")
    print()
    print("  A zero for a pipeline here is that failure mode firing on this query,")
    print("  not a broken pipeline: all three returned real, query-dependent nodes.")


def caveat(n_queries: int, document_id: str) -> None:
    print()
    print(RULE)
    print("  THIS IS NOT A RESULT")
    print(RULE)
    print(f"  {n_queries} queries on one filing ({document_id}) is an illustration of the")
    print("  mechanism, not a finding. The benchmark proper is 100 queries x 3")
    print("  pipelines x 3 K values = 900 runs, and it has not been run yet:")
    print("  the results table is empty.")
    print()
    print("  It does show the three paradigms disagreeing about which evidence")
    print("  answers the same question. That disagreement is what the benchmark")
    print("  is built to measure.")
    print(RULE)
    print()
