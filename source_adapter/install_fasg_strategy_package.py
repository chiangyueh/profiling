#!/usr/bin/env python3
"""Install one built FASG source-candidate package into an isolated OPP root.

The toolkit is never modified. The destination must already exist and must not
contain a vendors directory, so packages for different original strategies
cannot silently merge or overwrite each other. The official package installer
performs the copy into this explicitly supplied root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for part in iter(lambda: source.read(1024 * 1024), b""):
            value.update(part)
    return value.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-manifest", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path,
                        help="existing empty custom-OPP root, outside toolkit and repository")
    args = parser.parse_args()
    if not args.build_manifest.is_file():
        raise RuntimeError("missing build manifest: {}".format(args.build_manifest))
    manifest: dict[str, Any] = json.loads(args.build_manifest.read_text(encoding="utf-8"))
    package_text = manifest.get("source_package")
    if manifest.get("target") != "package" or manifest.get("source_tiling_observation_enabled") is not True or not package_text:
        raise RuntimeError("build manifest does not describe a completed package build")
    package = Path(package_text)
    if not package.is_file() or digest(package) != manifest.get("source_package_sha256"):
        raise RuntimeError("source package is absent or does not match its build manifest")
    if not args.destination.is_dir():
        raise RuntimeError("destination must be an existing directory: {}".format(args.destination))
    if (args.destination / "vendors").exists() or any(args.destination.iterdir()):
        raise RuntimeError("destination must be empty to preserve one-strategy isolation: {}".format(args.destination))
    result = subprocess.run([str(package), "--quiet", "--install-path=" + str(args.destination)],
                            capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError("official package installer failed rc={} output={}".format(
            result.returncode, (result.stdout + result.stderr).strip()))
    expected = args.destination / "vendors" / str(manifest["vendor"])
    if not expected.is_dir():
        raise RuntimeError("package installer completed without its isolated vendor directory")
    output = {
        "schema": "fasg_original_strategy_custom_opp_v1",
        "operator": manifest["operator"],
        "strategy_class": manifest["strategy_class"],
        "strategy_priority": manifest["strategy_priority"],
        "official_commit": manifest.get("official_commit"),
        "vendor": manifest["vendor"],
        "custom_opp_root": str(args.destination),
        "source_package": str(package),
        "source_package_sha256": digest(package),
        "source_tiling_observation_enabled": True,
        "overlay_manifest_sha256": manifest.get("overlay_manifest_sha256"),
        "toolkit_install_modified": False,
        "matmul_included": False,
    }
    output_path = args.destination / "fasg_source_candidate_package.json"
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except RuntimeError as error:
        print("source-candidate install error: {}".format(error), file=sys.stderr)
        sys.exit(2)
