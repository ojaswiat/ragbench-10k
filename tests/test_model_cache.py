"""Guards the durable fastembed model cache.

The default cache lives under the platform temp directory, which macOS
purges; a truncated download there leaves a populated snapshot directory
that fastembed treats as a cache hit and then fails to load from.
"""
from __future__ import annotations

import ast
import pathlib

from core import model_cache


def test_default_is_outside_the_repo_and_outside_temp():
    """1.2 GB of weights must never be committable, and never sit in a
    directory the OS is free to purge mid-run."""
    default = model_cache.DEFAULT_MODEL_CACHE_DIR
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    assert not str(default).startswith(str(repo_root)), default
    assert "/var/folders" not in str(default), default
    assert str(default).startswith(str(pathlib.Path.home())), default


def test_env_override_is_honoured_and_expanded(monkeypatch, tmp_path):
    monkeypatch.setenv("RAG_MODEL_CACHE_DIR", str(tmp_path / "weights"))
    assert model_cache.model_cache_dir() == str(tmp_path / "weights")


def test_directory_is_created_if_absent(monkeypatch, tmp_path):
    """fastembed does not create a missing cache_dir for you."""
    target = tmp_path / "does" / "not" / "exist"
    monkeypatch.setenv("RAG_MODEL_CACHE_DIR", str(target))
    assert not target.exists()
    model_cache.model_cache_dir()
    assert target.is_dir()


# Every fastembed model class that downloads weights. A new one added
# without a cache_dir would silently fall back to the temp directory.
_FASTEMBED_CLASSES = {"TextCrossEncoder", "TextEmbedding", "FastEmbedEmbedding"}

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _fastembed_constructions(path: pathlib.Path):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in _FASTEMBED_CLASSES:
                yield node


def test_every_fastembed_construction_pins_the_cache_dir():
    """An invariant over the tree, not a hardcoded list of call sites.

    The reranker cache broke because nothing asserted this; naming the
    invariant rather than the files means the next call site is checked by
    construction instead of by a reviewer remembering.
    """
    unpinned = []
    for path in _PROJECT_ROOT.rglob("*.py"):
        parts = set(path.parts)
        if ".venv" in parts or "tests" in parts or "backups" in parts:
            continue
        for node in _fastembed_constructions(path):
            if not any(keyword.arg == "cache_dir" for keyword in node.keywords):
                unpinned.append(f"{path.relative_to(_PROJECT_ROOT)}:{node.lineno} {node.func.id}")
    assert not unpinned, (
        "these fastembed models would download into the purgeable temp cache: "
        + ", ".join(unpinned)
    )
