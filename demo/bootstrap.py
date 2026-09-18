"""Seeds the demo's inputs so it runs from a clean checkout.

Two of the three pipelines can build their own index offline, but P3 cannot:
its summary tree is LLM-built over every node in the filing, and there is no
way to regenerate it without spending quota. The node store is the same kind
of problem, since it comes out of a paid parse of the source filing.

So both are shipped for the one demo filing and copied into place here. The
bundle covers MSFT_2023 only, which keeps it to a couple of megabytes and is
the filing the demo is pinned to. Everything else the demo needs, the P1 and
P2 indexes, builds locally on first run.

Nothing is overwritten. If a real benchmark.db is already present, this does
nothing at all and the demo reads whatever the full pipeline produced.
"""
import shutil
import sqlite3
from pathlib import Path

import pipelines.structural.build_summary_index as tree_builder

from demo.config import DB_PATH, DEMO_DOCUMENT

ASSETS = Path(__file__).resolve().parent / "assets"
BUNDLED_DB = ASSETS / "demo.db"
BUNDLED_TREE = ASSETS / "summary_index"


def _document_has_nodes(db_path: str, document_id: str) -> bool:
    """False for a missing file, an empty file, or a database without the table."""
    if not Path(db_path).exists():
        return False
    try:
        with sqlite3.connect(db_path) as connection:
            (count,) = connection.execute(
                "SELECT COUNT(*) FROM nodes WHERE document_id = ?", (document_id,)
            ).fetchone()
        return count > 0
    except sqlite3.OperationalError:
        return False


def ensure_demo_data(document_id: str = DEMO_DOCUMENT, log=print) -> dict:
    """Copies the bundled node store and summary tree into place if absent."""
    report = {}

    if _document_has_nodes(DB_PATH, document_id):
        report["nodes"] = "already present"
    elif document_id == DEMO_DOCUMENT and BUNDLED_DB.exists():
        Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(BUNDLED_DB, DB_PATH)
        log(f"  seeded benchmark.db from demo/assets/demo.db ({document_id})")
        report["nodes"] = "seeded from bundle"
    else:
        report["nodes"] = "missing, and no bundle for this filing"

    tree_destination = tree_builder.STORAGE_ROOT / document_id
    bundled_tree = BUNDLED_TREE / document_id
    if tree_destination.exists():
        report["summary_tree"] = "already present"
    elif bundled_tree.exists():
        tree_destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(bundled_tree, tree_destination)
        log(f"  seeded {tree_destination} from demo/assets/summary_index")
        report["summary_tree"] = "seeded from bundle"
    else:
        report["summary_tree"] = "missing, and no bundle for this filing"

    return report


if __name__ == "__main__":
    for key, value in ensure_demo_data().items():
        print(f"{key}: {value}")
