"""Fixed settings for the demo run.

The demo is pinned to one filing so it stays runnable on a laptop with no
API quota: P1 and P2 build their indexes per document, and MSFT_2023 is the
only filing that carries a query in all four difficulty quadrants.
"""
from pathlib import Path

DEMO_DOCUMENT = "MSFT_2023"
K = 3
QUADRANTS = (
    "Q1_Direct_Text",
    "Q2_Implicit_Text",
    "Q3_Direct_Table",
    "Q4_Implicit_Table",
)
PIPELINES = ("P1_vector", "P2_bm25", "P3_structural")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = str(PROJECT_ROOT / "benchmark.db")

RULE = "=" * 78
THIN = "-" * 78
