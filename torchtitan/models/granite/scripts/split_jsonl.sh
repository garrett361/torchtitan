#!/usr/bin/env bash
# Split large JSONL files into chunks of at most 1 GiB, splitting at line
# boundaries. Replaces originals after verification.
#
# Limitation: split -C will break lines that individually exceed 1 GiB into
# multiple chunks, producing invalid JSONL for those records. At 1 GiB this is
# effectively impossible for SFT chat data (typical large records are tens of MB).
#
# Usage:
#   ./split_jsonl.sh /path/to/dataset
#   DRY_RUN=1 ./split_jsonl.sh /path/to/dataset

set -euo pipefail

ROOT_DIR="${1:?Usage: $0 ROOT_DIR}"
MAX_SIZE="1G"
DRY_RUN="${DRY_RUN:-}"

max_bytes=$((1024 * 1024 * 1024))  # 1 GiB

if [[ ! -d "$ROOT_DIR" ]]; then
    echo "Error: not a directory: $ROOT_DIR" >&2
    exit 1
fi

split_count=0
chunk_count=0

# Cleanup partial chunks on failure
_current_dir=""
_current_base=""
cleanup_on_error() {
    if [[ -n "$_current_dir" && -n "$_current_base" ]]; then
        echo "Cleaning up partial chunks for ${_current_base}..." >&2
        rm -f "${_current_dir}/${_current_base}_part"*.jsonl
    fi
}
trap cleanup_on_error ERR

while IFS= read -r -d '' filepath; do
    # Skip symlinks
    if [[ -L "$filepath" ]]; then
        continue
    fi

    # Skip files that look like prior chunks (stem ends in _part<digits>)
    stem="${filepath%.jsonl}"
    if [[ "$stem" =~ (.*)_part[0-9]+$ ]]; then
        parent_stem="${BASH_REMATCH[1]}"
        if [[ -f "${parent_stem}.jsonl" ]]; then
            continue
        fi
    fi

    file_size=$(stat --format='%s' "$filepath")
    if (( file_size <= max_bytes )); then
        continue
    fi

    dir=$(dirname "$filepath")
    base=$(basename "$filepath" .jsonl)

    if [[ -n "$DRY_RUN" ]]; then
        est_chunks=$(( (file_size + max_bytes - 1) / max_bytes ))
        echo "[dry run] $filepath ($(numfmt --to=iec-i "$file_size")) -> at least $est_chunks chunks"
        split_count=$((split_count + 1))
        chunk_count=$((chunk_count + est_chunks))
        continue
    fi

    # Check for existing chunks (interrupted prior run)
    if compgen -G "${dir}/${base}_part*" > /dev/null 2>&1; then
        echo "Error: chunk files already exist for $filepath. Clean up and retry." >&2
        _current_dir=""
        _current_base=""
        exit 1
    fi

    # Track current file for cleanup trap
    _current_dir="$dir"
    _current_base="$base"

    echo "Splitting $filepath ($(numfmt --to=iec-i "$file_size"))..."

    # split -C: at most 1G per output file, split at line boundaries
    # -d: numeric suffixes, -a 3: support up to 1000 chunks
    split -C "$MAX_SIZE" -d -a 3 --additional-suffix=.jsonl \
        "$filepath" "${dir}/${base}_part"

    # Verify: sum chunk sizes via stat (avoids re-reading file contents)
    chunks_total=0
    n_chunks=0
    for chunk in "${dir}/${base}_part"*.jsonl; do
        [[ -f "$chunk" ]] || { echo "Error: split produced no output for $filepath" >&2; exit 1; }
        chunks_total=$(( chunks_total + $(stat --format='%s' "$chunk") ))
        n_chunks=$((n_chunks + 1))
    done

    if (( chunks_total != file_size )); then
        echo "Error: byte count mismatch after splitting $filepath" >&2
        echo "  original=$file_size, chunks_total=$chunks_total" >&2
        exit 1
    fi

    rm "$filepath"
    _current_dir=""
    _current_base=""
    echo "  -> $n_chunks chunks"

    split_count=$((split_count + 1))
    chunk_count=$((chunk_count + n_chunks))
done < <(find "$ROOT_DIR" -name '*.jsonl' -type f -print0 | sort -z)

echo "Done. $split_count files split into $chunk_count total chunks."
