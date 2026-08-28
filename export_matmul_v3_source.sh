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
import hashlib
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

def is_skipped_directory(name):
    upper = name.upper()
    lower = name.lower()
    return (
        name in skip_directories
        or upper.startswith("OPPROF")
        or lower.startswith("msprof")
        or lower.startswith("prof_")
    )

paths = []
for current, directories, files in os.walk(source_root, topdown=True, followlinks=False):
    directories[:] = sorted(name for name in directories if not is_skipped_directory(name))
    current_path = pathlib.Path(current)
    for name in sorted(files):
        path = current_path / name
        if path.is_symlink() or name in source_names or path.suffix.lower() in source_suffixes:
            paths.append(path)

snapshots = []
transient_missing = []
for path in sorted(paths):
    relative = path.relative_to(source_root).as_posix()
    try:
        if path.is_symlink():
            snapshots.append(("symlink", relative, os.readlink(path), None, None))
            continue
        file_stat = path.stat()
        data = path.read_bytes()
    except FileNotFoundError:
        transient_missing.append(relative)
        continue
    snapshots.append((
        "file",
        relative,
        data,
        format(stat.S_IMODE(file_stat.st_mode), "04o"),
        hashlib.sha256(data).hexdigest(),
    ))

file_count = sum(entry[0] == "file" for entry in snapshots)
symlink_count = sum(entry[0] == "symlink" for entry in snapshots)

with output_path.open("wb") as output:
    output.write(b"MATMUL_V3 SOURCE EXPORT\n")
    output.write(f"source_root={source_root}\n".encode("utf-8"))
    output.write(f"source_files={file_count}\n".encode("ascii"))
    output.write(f"symlinks={symlink_count}\n".encode("ascii"))
    output.write(f"transient_missing={len(transient_missing)}\n\n".encode("ascii"))

    for kind, relative, content, mode, sha256 in snapshots:
        if kind == "symlink":
            output.write(f"===== SYMLINK: {relative} =====\n".encode("utf-8"))
            output.write(f"target={content}\n".encode("utf-8"))
            output.write(f"===== END SYMLINK: {relative} =====\n\n".encode("utf-8"))
            continue

        output.write(
            f"===== BEGIN FILE: {relative} | mode={mode} | size={len(content)} | sha256={sha256} =====\n".encode("utf-8")
        )
        output.write(content)
        if not content.endswith(b"\n"):
            output.write(b"\n")
        output.write(f"===== END FILE: {relative} =====\n\n".encode("utf-8"))

    for relative in transient_missing:
        output.write(f"===== TRANSIENT FILE DISAPPEARED: {relative} =====\n".encode("utf-8"))
PY

ACTUAL_BYTES="$(stat -c '%s' "$TEMP_FILE")"
if (( ACTUAL_BYTES > MAX_BYTES )); then
    printf 'MATMUL_V3_SOURCE_EXPORT failed log_bytes=%s limit_bytes=%s\n' "$ACTUAL_BYTES" "$MAX_BYTES"
    exit 1
fi

mv -f "$TEMP_FILE" "$LOG_FILE"
trap - EXIT
printf 'MATMUL_V3_SOURCE_EXPORT passed bytes=%s log=%s\n' "$ACTUAL_BYTES" "$LOG_FILE"
