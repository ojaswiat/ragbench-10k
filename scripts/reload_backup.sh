#!/usr/bin/env bash
# Restores a snapshot created by backup_data.sh back into this project.
# Hardcoded to this project's layout -- not meant to be reused elsewhere.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BACKUPS_DIR="$PROJECT_ROOT/backups"

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

if [ ! -d "$BACKUPS_DIR" ]; then
    echo "No data found in the backups folder, exiting."
    exit 1
fi

BACKUP_DIRS=()
while IFS= read -r d; do
    BACKUP_DIRS+=("$d")
done < <(find "$BACKUPS_DIR" -maxdepth 1 -type d -name 'backup-*' -print | sort -r)

if [ ${#BACKUP_DIRS[@]} -eq 0 ]; then
    echo "No data found in the backups folder, exiting."
    exit 1
fi

echo "Available backups:"
i=1
for d in "${BACKUP_DIRS[@]}"; do
    ts="$(basename "$d" | sed 's/^backup-//')"
    date_part="${ts%%-*}"
    time_part="${ts##*-}"
    human="$(date -j -f "%Y%m%d%H%M%S" "${date_part}${time_part}" "+%A, %d %B, %Y - %I:%M %p" 2>/dev/null || echo "$ts")"
    echo "  $i) $human  ($(basename "$d"))"
    i=$((i+1))
done

echo ""
read -rp "Choose a backup number to restore: " CHOICE

if ! [[ "$CHOICE" =~ ^[0-9]+$ ]] || [ "$CHOICE" -lt 1 ] || [ "$CHOICE" -gt "${#BACKUP_DIRS[@]}" ]; then
    echo "Invalid choice. Exiting."
    exit 1
fi

SELECTED="${BACKUP_DIRS[$((CHOICE-1))]}"
echo ""
echo "Selected: $(basename "$SELECTED")"
echo "This overwrites current data at: ${PATHS[*]}"
read -rp "Continue? (y/n): " CONFIRM

if [ "$CONFIRM" != "y" ]; then
    echo "Aborted."
    exit 0
fi

echo ""
echo "Snapshotting current state first (safety net)..."
"$SCRIPT_DIR/backup_data.sh"

REPLACED=()
NEW=()
SAME=()

for rel in "${PATHS[@]}"; do
    src="$SELECTED/$rel"
    dest="$PROJECT_ROOT/$rel"
    [ -e "$src" ] || continue

    if [ ! -e "$dest" ]; then
        NEW+=("$rel")
    elif diff -rq "$src" "$dest" >/dev/null 2>&1; then
        SAME+=("$rel")
    else
        REPLACED+=("$rel")
    fi

    dest_parent="$PROJECT_ROOT/$(dirname "$rel")"
    mkdir -p "$dest_parent"
    rm -rf "$dest"
    cp -R "$src" "$dest_parent/"
done

echo ""
echo "Verifying restored files (checksum compare)..."
CORRUPT=()

for rel in "${PATHS[@]}"; do
    src="$SELECTED/$rel"
    dest="$PROJECT_ROOT/$rel"
    [ -e "$src" ] || continue

    if [ -d "$src" ]; then
        while IFS= read -r -d '' f; do
            relf="${f#"$src"/}"
            src_sum="$(shasum -a 256 "$f" | cut -d' ' -f1)"
            dest_sum="$(shasum -a 256 "$dest/$relf" 2>/dev/null | cut -d' ' -f1 || echo "MISSING")"
            if [ "$src_sum" != "$dest_sum" ]; then
                CORRUPT+=("$rel/$relf")
            fi
        done < <(find "$src" -type f -print0)
    else
        src_sum="$(shasum -a 256 "$src" | cut -d' ' -f1)"
        dest_sum="$(shasum -a 256 "$dest" 2>/dev/null | cut -d' ' -f1 || echo "MISSING")"
        if [ "$src_sum" != "$dest_sum" ]; then
            CORRUPT+=("$rel")
        fi
    fi
done

echo ""
echo "Restore summary:"
echo "  Replaced (${#REPLACED[@]}): ${REPLACED[*]:-none}"
echo "  New (${#NEW[@]}): ${NEW[*]:-none}"
echo "  Same/unchanged (${#SAME[@]}): ${SAME[*]:-none}"
if [ ${#CORRUPT[@]} -gt 0 ]; then
    echo "  CORRUPTION DETECTED (${#CORRUPT[@]}):"
    for c in "${CORRUPT[@]}"; do echo "    - $c"; done
    echo ""
    echo "Restore completed with checksum mismatches -- see above."
    exit 1
else
    echo "  Corruption check: none detected, all checksums match"
fi

echo ""
echo "Restore complete from $(basename "$SELECTED")."
