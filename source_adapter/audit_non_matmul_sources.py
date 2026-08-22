#!/usr/bin/env python3
"""Read-only provenance audit for non-MatMul original-strategy collection.

The audit neither downloads, compiles, launches, resets, nor writes to an NPU.
It verifies the exact source revisions and the finite strategy inventory that the
overlay generator is allowed to expose.
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
LOCK = json.loads((ROOT / "non_matmul_source_lock.json").read_text(encoding="utf-8"))


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_head(path: Path) -> str | None:
    if not (path / ".git").exists():
        return None
    result = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"],
                            capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def split_top_level(text: str) -> list[str]:
    depth = 0
    output: list[str] = []
    start = 0
    for index, char in enumerate(text):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif char == "," and depth == 0:
            output.append(text[start:index].strip())
            start = index + 1
    output.append(text[start:].strip())
    return output


def registrations(ophost: Path, macro: str, operator: str) -> list[dict[str, str]]:
    pattern = re.compile(
        re.escape(macro) + r"\s*\(\s*\"" + re.escape(operator) +
        r"\"\s*,\s*([A-Za-z_][A-Za-z0-9_]*)\s*,\s*([0-9]+)\s*\)",
        re.MULTILINE,
    )
    rows: list[dict[str, str]] = []
    for source in sorted(ophost.glob("*.cpp")):
        for match in pattern.finditer(source.read_text(encoding="utf-8", errors="replace")):
            rows.append({
                "class": match.group(1),
                "priority": match.group(2),
                "source": str(source),
            })
    return rows


def source_row(source_name: str, root: Path) -> dict[str, Any]:
    expected = LOCK["sources"][source_name]
    head = git_head(root)
    return {
        "root": str(root),
        "expected_commit": expected.get("commit"),
        "head": head,
        "revision_ok": head == expected.get("commit"),
    }


def file_checks(base: Path, pinned: dict[str, str]) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for relative, expected in pinned.items():
        path = base / relative
        actual = sha256(path)
        rows[relative] = {
            "path": str(path),
            "expected_sha256": expected,
            "actual_sha256": actual,
            "ok": actual == expected,
        }
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cann-ops-adv-root", default=os.environ.get(
        "CANN_OPS_ADV_SOURCE", LOCK["sources"]["cann_ops_adv"]["local_cache_hint"]))
    parser.add_argument("--cann-ops-root", default=os.environ.get(
        "CANN_OPS_SOURCE", LOCK["sources"]["cann_ops"]["local_cache_hint"]))
    parser.add_argument("--extracted-root", default=os.environ.get(
        "CCE_EXTRACT_ROOT", LOCK["sources"]["extracted_installed_source"]["local_cache_hint"]))
    args = parser.parse_args()

    adv = Path(args.cann_ops_adv_root)
    ops = Path(args.cann_ops_root)
    extracted = Path(args.extracted_root)
    fasg = LOCK["operators"]["flash_attention_score_grad"]
    fasg_root = adv / fasg["relative_root"]
    fasg_inventory = registrations(fasg_root / "ophost", fasg["registration_macro"], fasg["registration_operator"])
    expected_count = fasg["expected_strategy_count"]
    report: dict[str, Any] = {
        "schema": LOCK["schema"],
        "read_only": True,
        "npu_calls": 0,
        "downloads": 0,
        "compilations": 0,
        "matmul_included": False,
        "sources": {
            "cann_ops_adv": source_row("cann_ops_adv", adv),
            "cann_ops": source_row("cann_ops", ops),
            "extracted_installed_source": {"root": str(extracted), "present": extracted.is_dir()},
        },
        "operators": {
            "flash_attention_score_grad": {
                "file_checks": file_checks(fasg_root, fasg["pinned_files"]),
                "registered_original_strategies": fasg_inventory,
                "registered_strategy_count": len(fasg_inventory),
                "expected_strategy_count": expected_count,
                "strategy_inventory_ok": len(fasg_inventory) == expected_count,
            },
            "fused_infer_attention_score": {
                "file_checks": file_checks(
                    adv / LOCK["operators"]["fused_infer_attention_score"]["relative_root"],
                    LOCK["operators"]["fused_infer_attention_score"]["pinned_files"],
                ),
                "native_dispatch": LOCK["operators"]["fused_infer_attention_score"]["native_dispatch"],
            },
            "scatter_elements_v2": {
                "file_checks": file_checks(
                    ops / LOCK["operators"]["scatter_elements_v2"]["relative_root"],
                    LOCK["operators"]["scatter_elements_v2"]["pinned_files"],
                ),
                "native_dispatch": LOCK["operators"]["scatter_elements_v2"]["native_dispatch"],
            },
            "gather_elements_v2": {
                "file_checks": file_checks(
                    extracted / LOCK["operators"]["gather_elements_v2"]["relative_root"],
                    LOCK["operators"]["gather_elements_v2"]["pinned_files"],
                ),
                "native_dispatch": LOCK["operators"]["gather_elements_v2"]["native_dispatch"],
            },
            "transpose": LOCK["operators"]["transpose"],
            "gather_v2": LOCK["operators"]["gather_v2"],
        },
    }
    relevant_ok = (
        report["sources"]["cann_ops_adv"]["revision_ok"]
        and report["sources"]["cann_ops"]["revision_ok"]
        and report["operators"]["flash_attention_score_grad"]["strategy_inventory_ok"]
        and all(row["ok"] for op in ("flash_attention_score_grad", "fused_infer_attention_score",
                                       "scatter_elements_v2", "gather_elements_v2")
                for row in report["operators"][op]["file_checks"].values())
    )
    report["ready_for_fasg_overlay_generation"] = relevant_ok
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if relevant_ok else 2


if __name__ == "__main__":
    sys.exit(main())
