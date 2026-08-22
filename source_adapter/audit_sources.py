#!/usr/bin/env python3
"""Read-only provenance and invariant audit for source-preserving tiling work.

It neither downloads source nor compiles, launches, resets, or touches an NPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
LOCK_PATH = ROOT / "source_lock.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_head(path: Path) -> tuple[str | None, str | None]:
    if not (path / ".git").exists():
        return None, "not_a_git_worktree"
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        return None, result.stderr.strip() or "git_rev_parse_failed"
    return result.stdout.strip(), None


def text_has_all(path: Path, values: list[str]) -> tuple[bool, list[str]]:
    if not path.is_file():
        return False, ["missing_file"]
    text = path.read_text(encoding="utf-8", errors="replace")
    missing = [value for value in values if value not in text]
    return not missing, missing


def source_report(lock: dict[str, Any], official_root: Path, extracted_root: Path) -> dict[str, Any]:
    official = lock["official_source"]
    report: dict[str, Any] = {
        "official_root": str(official_root),
        "official_expected_commit": official["commit"],
        "official_head": None,
        "official_ok": False,
        "operators": {},
    }
    head, error = git_head(official_root)
    report["official_head"] = head
    report["official_ok"] = head == official["commit"]
    if error:
        report["official_error"] = error

    for name, spec in lock["operators"].items():
        row: dict[str, Any] = {
            "source": spec["source"],
            "declared_status": spec["status"],
            "source_root": str((official_root if spec["source"] == "official_open" else extracted_root) / spec["relative_root"]),
        }
        source_root = Path(row["source_root"])
        row["source_present"] = source_root.is_dir()
        if "reason" in spec:
            row["reason"] = spec["reason"]
        report["operators"][name] = row

    matmul = report["operators"]["mat_mul_v3"]
    matmul_root = official_root / lock["operators"]["mat_mul_v3"]["relative_root"]
    base_cpp = matmul_root / "op_host/mat_mul_v3_base_tiling.cpp"
    common_h = matmul_root / "op_host/mat_mul_v3_common.h"
    base_h = matmul_root / "op_host/mat_mul_v3_base_tiling.h"
    source_cpp = matmul_root / "op_host/mat_mul_v3_tiling.cpp"
    tiling_base_h = official_root / "src/common/inc/tiling/tiling_base.h"
    required = {
        "base_cpp": (base_cpp, [
            "#define DO_CACL_TILING_ENABLE(func) if (func) { break; }",
            "case TilingCalcSelect::ALL:",
            "case TilingCalcSelect::BASE:",
            "case TilingCalcSelect::SINGLE_CORE_SPLIT_K:",
            "case TilingCalcSelect::DETERMINISTIC_SPLIT_K:",
        ]),
        "common_h": (common_h, [
            "ALL = 0",
            "BASE = 1",
            "SINGLE_CORE_SPLIT_K = 2",
            "DETERMINISTIC_SPLIT_K = 3",
        ]),
        "base_h": (base_h, [
            "MatmulV3BaseTiling(gert::TilingContext* context,",
            "TilingCalcSelect tilingSelect = TilingCalcSelect::BASE",
        ]),
        "tiling_cpp": (source_cpp, [
            'REGISTER_TILING_TEMPLATE("MatMulV3", MatmulV3BaseTiling, 0);',
        ]),
        "tiling_base_h": (tiling_base_h, [
            "ge::graphStatus DoTiling()",
        ]),
    }
    source_checks: dict[str, Any] = {}
    for label, (path, tokens) in required.items():
        token_ok, missing = text_has_all(path, tokens)
        try:
            relative_key = str(path.relative_to(matmul_root))
        except ValueError:
            relative_key = "../../common/inc/tiling/tiling_base.h"
        expected_hash = lock["operators"]["mat_mul_v3"]["pinned_files"].get(relative_key)
        actual_hash = sha256(path) if path.is_file() else None
        source_checks[label] = {
            "path": str(path),
            "tokens_ok": token_ok,
            "missing_tokens": missing,
            "sha256": actual_hash,
            "pinned_sha256": expected_hash,
            "hash_ok": actual_hash == expected_hash,
        }
    matmul["original_route_checks"] = source_checks
    matmul["eligible_to_build"] = bool(
        report["official_ok"]
        and all(check["tokens_ok"] and check["hash_ok"] for check in source_checks.values())
    )
    return report


def toolkit_report(toolkit_root: Path, expected_version: str) -> dict[str, Any]:
    header = toolkit_root / "include/version/cann_version.h"
    output: dict[str, Any] = {"root": str(toolkit_root), "header": str(header), "ok": False}
    if not header.is_file():
        output["error"] = "missing_cann_version_header"
        return output
    content = header.read_text(encoding="utf-8", errors="replace")
    match = re.search(r'#define\s+CANN_VERSION_STR\s+"([^"]+)"', content)
    output["detected_version"] = match.group(1) if match else None
    output["ok"] = output["detected_version"] == expected_version
    if not output["ok"]:
        output["error"] = "cann_version_mismatch"
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-root", default=os.environ.get(
        "CANN_OPS_SOURCE", "/usr/local/Ascend/.source_cache/cann-ops-8.1rc1"))
    parser.add_argument("--extracted-root", default=os.environ.get(
        "CCE_EXTRACT_ROOT", "/home/CCE_EXTRACT/ops_cce"))
    parser.add_argument("--toolkit-root", default=os.environ.get(
        "ASCEND_TOOLKIT_HOME", "/usr/local/Ascend/ascend-toolkit/latest"))
    args = parser.parse_args()

    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    report = {
        "schema": lock["schema"],
        "read_only": True,
        "npu_calls": 0,
        "downloads": 0,
        "compilations": 0,
        "invariants": lock["invariants"],
        "toolkit": toolkit_report(Path(args.toolkit_root), lock["target"]["cann_version"]),
        "sources": source_report(lock, Path(args.official_root), Path(args.extracted_root)),
    }
    report["ready_for_minimal_matmul_overlay"] = bool(
        report["toolkit"]["ok"] and report["sources"]["operators"]["mat_mul_v3"].get("eligible_to_build")
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["ready_for_minimal_matmul_overlay"] else 2


if __name__ == "__main__":
    sys.exit(main())
