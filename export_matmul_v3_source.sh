#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_DIR="${MATMUL_V3_SOURCE_DIR:-/home/spbgu_jointlab/matmul_v3}"
LOG_DIR="${MATMUL_V3_EXPORT_LOG_DIR:-$ROOT/results/matmul_v3_source_export/logs}"
LOG_FILE="$LOG_DIR/1.log"
MAX_BYTES=$((50 * 1024 * 1024))

[[ -d "$SOURCE_DIR" ]] || {
    printf 'MATMUL_V3_SOURCE_EXPORT failed source_directory_missing=%s\n' "$SOURCE_DIR"
    exit 1
}

mkdir -p "$LOG_DIR"
TEMP_FILE="$(mktemp "$LOG_DIR/.1.log.tmp.XXXXXX")"
trap 'rm -f "$TEMP_FILE"' EXIT

python3 - "$SOURCE_DIR" "$TEMP_FILE" <<'PY'
import base64
import hashlib
import json
import os
import pathlib
import stat
import sys

source_root = pathlib.Path(sys.argv[1]).resolve()
output_path = pathlib.Path(sys.argv[2])

source_suffixes = {
    ".c", ".cc", ".cpp", ".cxx",
    ".h", ".hh", ".hpp", ".hxx", ".inc",
    ".py", ".sh", ".bash",
    ".cmake", ".mk", ".in",
    ".json", ".ini", ".cfg", ".conf",
    ".yaml", ".yml", ".toml", ".proto", ".md", ".txt",
}
source_names = {"CMakeLists.txt", "Makefile", "Dockerfile"}
skip_directories = {
    ".git", "__pycache__", "input", "output", "profiler", "profiling",
    "build", "out", ".cache", ".pytest_cache",
}

source_files = []
metadata_files = []
symlinks = []

for current, directories, files in os.walk(source_root, topdown=True, followlinks=False):
    directories[:] = sorted(name for name in directories if name not in skip_directories)
    current_path = pathlib.Path(current)
    for name in sorted(files):
        path = current_path / name
        relative = path.relative_to(source_root).as_posix()
        if path.is_symlink():
            symlinks.append((relative, os.readlink(path)))
            continue
        if not path.is_file():
            continue
        if name in source_names or path.suffix.lower() in source_suffixes:
            source_files.append((relative, path))
        else:
            metadata_files.append((relative, path))

source_files.sort()
metadata_files.sort()
symlinks.sort()

def digest(path):
    hasher = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            hasher.update(chunk)
    return size, hasher.hexdigest()

with output_path.open("w", encoding="utf-8", newline="\n") as output:
    header = {
        "record_type": "source_export",
        "schema": "matmul_v3_source_export_v1",
        "source_root": str(source_root),
        "source_file_count": len(source_files),
        "metadata_file_count": len(metadata_files),
        "symlink_count": len(symlinks),
        "content_encoding": "base64",
    }
    output.write(json.dumps(header, sort_keys=True, separators=(",", ":")) + "\n")

    for relative, target in symlinks:
        row = {
            "record_type": "symlink",
            "path": relative,
            "target": target,
        }
        output.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")

    for relative, path in source_files:
        data = path.read_bytes()
        row = {
            "record_type": "source_file",
            "path": relative,
            "mode": format(stat.S_IMODE(path.stat().st_mode), "04o"),
            "size": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            "encoding": "base64",
            "content": base64.b64encode(data).decode("ascii"),
        }
        output.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")

    for relative, path in metadata_files:
        size, sha256 = digest(path)
        row = {
            "record_type": "non_source_file",
            "path": relative,
            "mode": format(stat.S_IMODE(path.stat().st_mode), "04o"),
            "size": size,
            "sha256": sha256,
        }
        output.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
PY

ACTUAL_BYTES="$(stat -c '%s' "$TEMP_FILE")"
if (( ACTUAL_BYTES > MAX_BYTES )); then
    printf 'MATMUL_V3_SOURCE_EXPORT failed log_bytes=%s limit_bytes=%s\n' "$ACTUAL_BYTES" "$MAX_BYTES"
    exit 1
fi

mv -f "$TEMP_FILE" "$LOG_FILE"
trap - EXIT
printf 'MATMUL_V3_SOURCE_EXPORT passed files=%s bytes=%s log=%s\n' \
    "$(wc -l < "$LOG_FILE")" "$ACTUAL_BYTES" "$LOG_FILE"
