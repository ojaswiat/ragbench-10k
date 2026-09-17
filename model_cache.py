"""Durable on-disk location for the fastembed ONNX models.

fastembed defaults to a cache under the platform temp directory. On macOS
that is /var/folders/.../T, which the OS purges. A purge *between* runs is
survivable, because the model simply downloads again. A purge or an
interrupted download *during* the 900-cell benchmark is not: P1 needs the
cross-encoder for all 300 of its cells, and a truncated download is worse
than an absent one, because it leaves a populated snapshot directory that
fastembed treats as a cache hit and then fails to load from.

Kept outside the repository so the ~1.2 GB of weights is never committed and
never swept into project/backups by scripts/backup_data.sh.
"""
import os
from pathlib import Path

_ENV_VAR = "RAG_MODEL_CACHE_DIR"
DEFAULT_MODEL_CACHE_DIR = Path.home() / ".cache" / "rag-techniques" / "fastembed"


def model_cache_dir() -> str:
    """The directory fastembed downloads and loads ONNX models from.

    Overridable through RAG_MODEL_CACHE_DIR so a machine with a small home
    volume, or a CI runner, can place the weights elsewhere without editing
    any call site.
    """
    override = os.environ.get(_ENV_VAR)
    path = Path(override).expanduser() if override else DEFAULT_MODEL_CACHE_DIR
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


__all__ = ["model_cache_dir", "DEFAULT_MODEL_CACHE_DIR"]
