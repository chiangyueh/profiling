#!/usr/bin/env python3
"""Create isolated original-strategy overlays for FlashAttentionScoreGrad.

Each overlay is a new git worktree at the pinned public source revision.  It
keeps exactly one original `REGISTER_TILING_TEMPLATE` registration and disables
the other registrations. No strategy class, predicate, kernel, or generated
tiling field is changed. The FASG entry has one audit-only status passthrough:
after the original dispatcher returns, it records the already-generated raw
tiling identity only when a temporary audit-path environment variable is set.
It never changes the context or return status. Therefore a failed overlay means
that the chosen original strategy rejected the original TilingContext; it is
not converted into a synthetic candidate.

This script only creates source overlays and JSON manifests.  It neither
downloads source nor compiles, launches, resets, or otherwise calls an NPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
LOCK = json.loads((ROOT / "non_matmul_source_lock.json").read_text(encoding="utf-8"))
OP = LOCK["operators"]["flash_attention_score_grad"]
SOURCE = LOCK["sources"]["cann_ops_adv"]
MACRO = OP["registration_macro"]
OPERATOR = OP["registration_operator"]
ENTRY_RELATIVE = Path(OP["relative_root"]) / "ophost/flash_attention_score_grad_tiling.cpp"
AUDIT_SENTINEL = "FASG_SOURCE_TILING_AUDIT_V1"


def instrument_entry_source(source: str) -> str:
    """Add an audit record while returning the original dispatcher status."""
    if AUDIT_SENTINEL in source:
        return source
    original = """ASCENDC_EXTERN_C ge::graphStatus TilingFlashAttentionGradScore(gert::TilingContext *context)
{
    if (CheckParams(context) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (IsEmptyOutput(context)) {
        FlashAttentionScoreGradTiling flashAttentionScoreGradTiling;
        return flashAttentionScoreGradTiling.RunEmptyTiling(context);
    } else {
        return TilingRegistry::GetInstance().DoTilingImpl(context);
    }
}"""
    replacement = r"""// FASG_SOURCE_TILING_AUDIT_V1: observational only; no tiling state is modified.
#include <cstdio>
#include <cstdlib>
#include <cstdint>

static uint64_t FASGSourceAuditHash(const uint8_t *data, size_t size)
{
    uint64_t value = 1469598103934665603ULL;
    for (size_t index = 0; index < size; ++index) { value ^= data[index]; value *= 1099511628211ULL; }
    return value;
}

static ge::graphStatus FASGSourceAuditResult(gert::TilingContext *context, ge::graphStatus status)
{
    const char *path = std::getenv("FASG_TILING_AUDIT_PATH");
    if (path == nullptr || context == nullptr) { return status; }
    auto *raw = context->GetRawTilingData();
    const size_t rawSize = raw == nullptr ? 0U : raw->GetDataSize();
    const auto *rawBytes = raw == nullptr ? nullptr : static_cast<const uint8_t *>(raw->GetData());
    const uint64_t digest = rawBytes == nullptr ? 0ULL : FASGSourceAuditHash(rawBytes, rawSize);
    std::FILE *output = std::fopen(path, "a");
    if (output != nullptr) {
        std::fprintf(output, "{\"schema\":\"fasg_raw_tiling_observation_v1\",\"status\":%d,\"tiling_key\":%llu,\"block_dim\":%u,\"raw_bytes\":%llu,\"raw_fnv1a64\":%llu}\n",
                     static_cast<int>(status), static_cast<unsigned long long>(context->GetTilingKey()),
                     context->GetBlockDim(), static_cast<unsigned long long>(rawSize),
                     static_cast<unsigned long long>(digest));
        std::fclose(output);
    }
    return status;
}

ASCENDC_EXTERN_C ge::graphStatus TilingFlashAttentionGradScore(gert::TilingContext *context)
{
    if (CheckParams(context) != ge::GRAPH_SUCCESS) { return FASGSourceAuditResult(context, ge::GRAPH_FAILED); }
    if (IsEmptyOutput(context)) {
        FlashAttentionScoreGradTiling flashAttentionScoreGradTiling;
        return FASGSourceAuditResult(context, flashAttentionScoreGradTiling.RunEmptyTiling(context));
    }
    return FASGSourceAuditResult(context, TilingRegistry::GetInstance().DoTilingImpl(context));
}"""
    if source.count(original) != 1:
        raise RuntimeError("cannot locate the exact original FASG tiling entry for instrumentation")
    return source.replace(original, replacement)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for part in iter(lambda: source.read(1024 * 1024), b""):
            value.update(part)
    return value.hexdigest()


def run(argv: list[str]) -> str:
    result = subprocess.run(argv, capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError("command failed rc={} argv={} stderr={}".format(
            result.returncode, " ".join(argv), result.stderr.strip()))
    return result.stdout.strip()


@dataclass(frozen=True)
class Registration:
    relative_path: Path
    start: int
    end: int
    source_class: str
    priority: int
    text: str


def find_close(text: str, opening: int) -> int:
    """Return the index one past `);` for a registration macro invocation."""
    depth = 0
    quote: str | None = None
    escaped = False
    for index in range(opening, len(text)):
        char = text[index]
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char in ("'", '\"'):
            quote = char
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                cursor = index + 1
                while cursor < len(text) and text[cursor].isspace():
                    cursor += 1
                if cursor >= len(text) or text[cursor] != ";":
                    raise RuntimeError("registration macro is missing trailing semicolon")
                return cursor + 1
    raise RuntimeError("unterminated registration macro")


def registrations(source_root: Path) -> list[Registration]:
    op_root = source_root / OP["relative_root"] / "ophost"
    start_pattern = re.compile(re.escape(MACRO) + r"\s*\(")
    class_pattern = re.compile(
        r"^" + re.escape(MACRO) + r"\s*\(\s*\"" + re.escape(OPERATOR) +
        r"\"\s*,\s*([A-Za-z_][A-Za-z0-9_]*)\s*,\s*([0-9]+)\s*\);$",
        re.DOTALL,
    )
    output: list[Registration] = []
    for path in sorted(op_root.glob("*.cpp")):
        content = path.read_text(encoding="utf-8")
        for match in start_pattern.finditer(content):
            end = find_close(content, content.find("(", match.start()))
            invocation = content[match.start():end]
            parsed = class_pattern.match(invocation.strip())
            if not parsed:
                continue
            output.append(Registration(
                relative_path=path.relative_to(source_root),
                start=match.start(), end=end,
                source_class=parsed.group(1), priority=int(parsed.group(2)), text=invocation,
            ))
    output.sort(key=lambda row: (row.priority, row.source_class), reverse=True)
    return output


def require_pinned_source(source_root: Path) -> list[Registration]:
    head = run(["git", "-C", str(source_root), "rev-parse", "HEAD"])
    if head != SOURCE["commit"]:
        raise RuntimeError("source revision mismatch: expected={} actual={}".format(SOURCE["commit"], head))
    status = run(["git", "-C", str(source_root), "status", "--porcelain", "--untracked-files=no"])
    if status:
        raise RuntimeError("source worktree is modified; refuse to derive an overlay from it")
    for relative, expected in OP["pinned_files"].items():
        actual = digest(source_root / OP["relative_root"] / relative)
        if actual != expected:
            raise RuntimeError("pinned source hash mismatch: {}".format(relative))
    rows = registrations(source_root)
    if len(rows) != OP["expected_strategy_count"]:
        raise RuntimeError("unexpected original strategy count: expected={} actual={}".format(
            OP["expected_strategy_count"], len(rows)))
    return rows


def overlay_source(text: str, rows: list[Registration], selected: Registration, relative: Path) -> str:
    local = [row for row in rows if row.relative_path == relative]
    for row in sorted(local, key=lambda item: item.start, reverse=True):
        if row == selected:
            continue
        replacement = "/* source-candidate collector disabled registration: {} priority={} */".format(
            row.source_class, row.priority)
        text = text[:row.start] + replacement + text[row.end:]
    return text


def allowed_overlay_source(before: str, rows: list[Registration], selected: Registration,
                           relative: Path) -> str:
    """Return the only permitted source content for one overlay file."""
    after = overlay_source(before, rows, selected, relative)
    if relative == ENTRY_RELATIVE:
        after = instrument_entry_source(after)
    return after


def existing_overlay(source_root: Path, output: Path, selected: Registration,
                     rows: list[Registration]) -> dict[str, object] | None:
    """Return a verified existing overlay, or None when it has not been made.

    This makes a stopped source-preparation stage resumable without accepting a
    partially edited worktree. `version.info` is deliberately not checked here:
    the package builder may have made its separately attested compatibility-only
    metadata edit there. Every registration source plus the one audit entry
    source must still equal the allowed collector transformation exactly.
    """
    if not output.exists():
        return None
    manifest_path = output / "source_candidate_overlay.json"
    if not manifest_path.is_file():
        raise RuntimeError("existing overlay has no provenance manifest: {}".format(output))
    metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = {
        "official_commit": SOURCE["commit"],
        "strategy_class": selected.source_class,
        "strategy_priority": selected.priority,
        "algorithm_source_changes": False,
    }
    if any(metadata.get(key) != value for key, value in required.items()):
        raise RuntimeError("existing overlay provenance does not match requested strategy: {}".format(output))
    affected = {row.relative_path for row in rows} | {ENTRY_RELATIVE}
    for relative in sorted(affected):
        expected = allowed_overlay_source((source_root / relative).read_text(encoding="utf-8"), rows, selected, relative)
        actual = (output / relative).read_text(encoding="utf-8")
        if actual != expected:
            raise RuntimeError("existing overlay source differs outside the allowed registration isolation: {}".format(
                output / relative))
    metadata["resumed_existing_overlay"] = True
    return metadata


def write_overlay(source_root: Path, output_parent: Path, selected: Registration, rows: list[Registration]) -> dict[str, object]:
    output = output_parent / ("fasg_" + selected.source_class.lower())
    existing = existing_overlay(source_root, output, selected, rows)
    if existing is not None:
        return existing
    run(["git", "-C", str(source_root), "worktree", "add", "--detach", str(output), SOURCE["commit"]])
    changed: list[dict[str, str]] = []
    affected = {row.relative_path for row in rows} | {ENTRY_RELATIVE}
    for relative in sorted(affected):
        target = output / relative
        before = target.read_text(encoding="utf-8")
        after = allowed_overlay_source(before, rows, selected, relative)
        if before != after:
            target.write_text(after, encoding="utf-8")
            changed.append({
                "path": str(relative),
                "kind": "source_tiling_observation" if relative == ENTRY_RELATIVE else "registration_isolation",
                "sha256_before": hashlib.sha256(before.encode("utf-8")).hexdigest(),
                "sha256_after": digest(target),
            })
    status = run(["git", "-C", str(output), "status", "--porcelain", "--untracked-files=no"])
    metadata: dict[str, object] = {
        "schema": "fasg_original_strategy_overlay_v1",
        "operator": OPERATOR,
        "official_url": SOURCE["url"],
        "official_tag": SOURCE["tag"],
        "official_commit": SOURCE["commit"],
        "strategy_class": selected.source_class,
        "strategy_priority": selected.priority,
        "enabled_registration": selected.text,
        "disabled_original_registrations": [
            {"class": row.source_class, "priority": row.priority}
            for row in rows if row != selected
        ],
        "overlay": str(output),
        "source_status": status.splitlines(),
        "modified_registration_files": changed,
        "instrumentation": {
            "enabled": True,
            "scope": "FASG entry return status plus already-generated raw tiling key/blockDim/bytes/digest",
            "mechanism": "write one JSON line only when FASG_TILING_AUDIT_PATH is supplied",
            "mutates_tiling_context": False,
        },
        "algorithm_source_changes": False,
        "candidate_rule": "run exactly this original strategy on an unchanged TilingContext; deduplicate only exact observed raw tiling identities and retain only later output-validated executions",
        "forbidden": LOCK["collection_contract"]["forbidden"],
    }
    (output / "source_candidate_overlay.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-parent", required=True, type=Path,
                        help="existing directory outside this repository")
    parser.add_argument("--strategy", action="append", default=[],
                        help="original strategy class to prepare; repeatable (default: all)")
    args = parser.parse_args()
    if not args.output_parent.is_dir():
        raise RuntimeError("output parent does not exist: {}".format(args.output_parent))
    rows = require_pinned_source(args.source_root)
    allowed = {row.source_class for row in rows}
    requested = set(args.strategy) if args.strategy else allowed
    unknown = sorted(requested - allowed)
    if unknown:
        raise RuntimeError("unknown original strategy: {}".format(", ".join(unknown)))
    prepared = [write_overlay(args.source_root, args.output_parent, row, rows)
                for row in rows if row.source_class in requested]
    print(json.dumps({
        "schema": "fasg_original_strategy_overlay_batch_v1",
        "matmul_included": False,
        "prepared_count": len(prepared),
        "original_registered_strategy_count": len(rows),
        "overlays": prepared,
    }, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except RuntimeError as error:
        print("source-candidate error: {}".format(error), file=sys.stderr)
        sys.exit(2)
