#!/usr/bin/env bash
# Snapshots this project's non-portable, non-git-tracked data into
# backups/backup-<timestamp>/ preserving the original relative paths.
# Hardcoded to this project's layout -- not meant to be reused elsewhere.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BACKUPS_DIR="$PROJECT_ROOT/backups"

TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
DEST="$BACKUPS_DIR/backup-$TIMESTAMP"

PATHS=(
    "benchmark.db"
    "benchmark.db-wal"
    "benchmark.db-shm"
    "storage"
    ".env"
    "data/raw"
    "data/parsed"
    "logs"
)

mkdir -p "$DEST"

BACKED_UP=()
SKIPPED=()

for rel in "${PATHS[@]}"; do
    src="$PROJECT_ROOT/$rel"
    if [ -e "$src" ]; then
        target_parent="$DEST/$(dirname "$rel")"
        mkdir -p "$target_parent"
        cp -R "$src" "$target_parent/"
        BACKED_UP+=("$rel")
    else
        SKIPPED+=("$rel")
    fi
done

if [ ${#BACKED_UP[@]} -eq 0 ]; then
    echo "Nothing to back up -- no source paths exist. Removing empty snapshot."
    rmdir "$DEST" 2>/dev/null || true
    exit 1
fi

TOTAL_SIZE="$(du -sh "$DEST" | cut -f1)"

echo "Backed up (${#BACKED_UP[@]}):"
for p in "${BACKED_UP[@]}"; do echo "  - $p"; done

if [ ${#SKIPPED[@]} -gt 0 ]; then
    echo ""
    echo "Skipped (not found, ${#SKIPPED[@]}):"
    for p in "${SKIPPED[@]}"; do echo "  - $p"; done
fi

echo ""
echo "Snapshot size: $TOTAL_SIZE"
echo "Location: $DEST"
