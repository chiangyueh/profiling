#!/usr/bin/env python3
"""Create an isolated, native-CANN GatherElements dynamic-source overlay.

This deliberately uses the GatherElements implementation shipped by the
*same* CANN 8.1 installation as the reference. It creates a complete private
OPP root whose ``built-in`` tree remains a read-only link to CANN, and whose
``vendors/config.ini`` gives the source overlay priority for the already
registered ``GatherElements`` type. The implementation and tiling body stay
original. No installed file is edited.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any


AUDIT_SCHEMA = "gather_elements_native_dynamic_source_observation_v1"
AUDIT_ENV = "GATHER_ELEMENTS_TILING_AUDIT_PATH"
CORE_ENV = "GATHER_ELEMENTS_SOURCE_AIV_CAP"
UB_ENV = "GATHER_ELEMENTS_SOURCE_UB_DIVISOR"
DISPATCH_ENV = "GATHER_ELEMENTS_SOURCE_DISPATCH"
VENDOR = "source_gather_elements"
SOURCE_OPERATOR_TYPE = "GatherElements"
SOURCE_MODULE = "gather_elements"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def paths(cann: Path) -> tuple[Path, Path, Path]:
    tbe = cann / "opp" / "built-in" / "op_impl" / "ai_core" / "tbe"
    source = tbe / "impl" / "dynamic" / "gather_elements.py"
    config = tbe / "config" / "ascend910b" / "aic-ascend910b-ops-info.json"
    if not source.is_file() or not config.is_file():
        raise RuntimeError("installed CANN 8.1 GatherElements dynamic source is unavailable")
    return tbe, source, config


def instrumentation(source: str) -> str:
    sentinel = "GATHER_ELEMENTS_NATIVE_DYNAMIC_SOURCE_AUDIT_V1"
    if sentinel in source:
        return source
    imports = "from tbe.tik.common.tik_get_soc_name import get_soc_name\n"
    prelude = r'''from tbe.tik.common.tik_get_soc_name import get_soc_name
import hashlib as _ge_hashlib
import json as _ge_json
import os as _ge_os

# GATHER_ELEMENTS_NATIVE_DYNAMIC_SOURCE_AUDIT_V1.  This source is executed by
# CANN's normal dynamic compiler; the audit observes the exact bounded source
# configuration after a successful BuildCCE and never edits flow-table data.
def _ge_read_cap(name, allowed, current):
    raw = _ge_os.environ.get(name)
    if raw is None:
        return current
    try:
        value = int(raw)
    except ValueError:
        raise RuntimeError("invalid %s" % name)
    if value not in allowed or value > current:
        raise RuntimeError("invalid %s" % name)
    return value

def _ge_visible_ub_size():
    current = tbe_platform_adapter.get_soc_spec(tbe_platform_adapter.UB_SIZE)
    divisor = _ge_read_cap("GATHER_ELEMENTS_SOURCE_UB_DIVISOR", (1, 2, 4, 8), 1)
    return current // divisor

def _ge_emit_import_audit():
    # This executes before the source entrypoint. It distinguishes a private
    # source never selected from one selected but failing in native BuildCCE.
    target = _ge_os.environ.get("GATHER_ELEMENTS_TILING_AUDIT_PATH")
    if not target:
        return
    row = {
        "schema": "gather_elements_native_dynamic_source_observation_v1",
        "event": "module_imported",
        "operator_type": "GatherElements",
        "source_file": _ge_os.path.realpath(__file__),
        "pid": _ge_os.getpid(),
        "dispatch": _ge_os.environ.get("GATHER_ELEMENTS_SOURCE_DISPATCH"),
        "ascend_opp_path": _ge_os.environ.get("ASCEND_OPP_PATH"),
        "ascend_custom_opp_path": _ge_os.environ.get("ASCEND_CUSTOM_OPP_PATH"),
    }
    with open(target, "a", encoding="utf-8") as handle:
        handle.write(_ge_json.dumps(row, sort_keys=True, separators=(",", ":")) + "\\n")

def _ge_emit_audit(obj, x_dict, indices_dict, dim):
    target = _ge_os.environ.get("GATHER_ELEMENTS_TILING_AUDIT_PATH")
    if not target:
        return
    identity = {
        "source_sha256": "__SOURCE_SHA256__",
        "aiv_core_cap": obj.core_num,
        "ub_cap_divisor": obj._ge_ub_divisor,
        "shape": list(x_dict.get("shape", ())),
        "index_shape": list(indices_dict.get("shape", ())),
        "dtype": str(x_dict.get("dtype", "")).lower(),
        "index_dtype": str(indices_dict.get("dtype", "")).lower(),
        "axis": int(dim),
    }
    encoded = _ge_json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    row = dict(identity)
    row.update({"schema": "gather_elements_native_dynamic_source_observation_v1",
                "event": "tiling_generated",
                "operator_type": "GatherElements",
                "status": 0,
                "source_variant_sha256": _ge_hashlib.sha256(encoded).hexdigest()})
    with open(target, "a", encoding="utf-8") as handle:
        handle.write(_ge_json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
_ge_emit_import_audit()
'''
    if source.count(imports) != 1:
        raise RuntimeError("cannot locate native GatherElements import anchor")
    source = source.replace(imports, prelude)
    branch_ub = "    ub_size = tbe_platform_adapter.get_soc_spec(tbe_platform_adapter.UB_SIZE)\n"
    if source.count(branch_ub) != 1:
        raise RuntimeError("cannot locate native GatherElements support-check UB anchor")
    source = source.replace(branch_ub, "    ub_size = _ge_visible_ub_size()\n")
    resources = '''        self.ub_size = tbe_platform_adapter.get_soc_spec(tbe_platform_adapter.UB_SIZE)
        self.core_num = tbe_platform_adapter.get_soc_spec(tbe_platform_adapter.CORE_NUM)'''
    replacement = r'''        self.ub_size = tbe_platform_adapter.get_soc_spec(tbe_platform_adapter.UB_SIZE)
        self.core_num = tbe_platform_adapter.get_soc_spec(tbe_platform_adapter.CORE_NUM)
        # Bounded legal source inputs.  They only reduce the hardware values
        # the original source publishes in compile_info.
        self.core_num = _ge_read_cap("GATHER_ELEMENTS_SOURCE_AIV_CAP", tuple(range(1, self.core_num + 1)), self.core_num)
        self._ge_ub_divisor = _ge_read_cap("GATHER_ELEMENTS_SOURCE_UB_DIVISOR", (1, 2, 4, 8), 1)
        self.ub_size = _ge_visible_ub_size()'''
    if source.count(resources) != 1:
        raise RuntimeError("cannot locate native GatherElements resource anchor")
    source = source.replace(resources, replacement)
    final = '''    obj = GatherElements(x_dict, indices_dict, y_dict, dim, kernel_name)
    return obj.gather_elements_compute()'''
    final_replacement = '''    obj = GatherElements(x_dict, indices_dict, y_dict, dim, kernel_name)
    result = obj.gather_elements_compute()
    _ge_emit_audit(obj, x_dict, indices_dict, dim)
    return result'''
    if source.count(final) != 1:
        raise RuntimeError("cannot locate native GatherElements entrypoint")
    return source.replace(final, final_replacement)


def expected(output: Path, cann: Path, source: Path, config: Path) -> dict[str, Any]:
    # CANN's OPP loader discovers vendor priority from
    # ``ASCEND_OPP_PATH/vendors/config.ini``.  A vendor path alone cannot
    # introduce a new op type without an op-proto library.  This root keeps
    # the registered built-in type and gives this source vendor priority.
    root = output / "runtime_opp"
    vendor_root = root / "vendors" / VENDOR
    impl_dir = VENDOR + "_impl"
    source_target = vendor_root / "op_impl" / "ai_core" / "tbe" / impl_dir / "dynamic" / (SOURCE_MODULE + ".py")
    return {
        "schema": "gather_elements_native_dynamic_overlay_v6",
        "operator": "GatherElements",
        "source_operator_type": SOURCE_OPERATOR_TYPE,
        "source_module": SOURCE_MODULE,
        "runtime_op": "gather_elements",
        "source_kind": "installed_cann81_native_dynamic_source",
        "cann_root": str(cann.resolve()),
        "cann_version_file_sha256": digest(cann / "opp" / "version.info"),
        "installed_source": str(source),
        "installed_source_sha256": digest(source),
        "installed_config": str(config),
        "installed_config_sha256": digest(config),
        "installed_opp_root": str(cann / "opp"),
        "runtime_opp_root": str(root),
        "runtime_opp_layout": {
            "built_in_symlink": str(root / "built-in"),
            "built_in_target": str(cann / "opp" / "built-in"),
            "vendor_priority_file": str(root / "vendors" / "config.ini"),
            "vendor_priority": VENDOR,
        },
        "vendor": VENDOR,
        "vendor_impl_directory": impl_dir,
        "vendor_root": str(vendor_root),
        "source_file": str(source_target),
        "instrumentation": {
            "enabled": True, "mutates_tiling_context": False,
            "audit_schema": AUDIT_SCHEMA, "audit_environment": AUDIT_ENV,
            "source_budget_environment": CORE_ENV,
            "dispatch_environment": DISPATCH_ENV,
            "dispatch_value": "aclop_compile_and_execute",
        },
        "hardware_envelope_heuristic": {
            "enabled": True, "environment": UB_ENV, "audit_field": "ub_cap_divisor",
            "resource": "source_visible_ub_capacity", "divisors": [2, 4, 8], "max_anchors": 16,
        },
        "strategy_algorithm_changes": False,
        "kernel_algorithm_changes": False,
        "formal_data_gate": "the registered GatherElements type must be selected from the private ASCEND_OPP_PATH vendor-priority overlay, launch through aclopCompileAndExecute, and exactly match an installed aclnnGather reference",
    }


def validate_existing(output: Path, planned: dict[str, Any], original: str) -> dict[str, Any] | None:
    manifest_path = output / "native_dynamic_overlay.json"
    if not output.exists():
        return None
    if not manifest_path.is_file():
        raise RuntimeError("incomplete private native GatherElements overlay exists")
    item = json.loads(manifest_path.read_text(encoding="utf-8"))
    for key in ("schema", "operator", "source_operator_type", "source_module", "runtime_op", "source_kind", "cann_root", "cann_version_file_sha256",
                "installed_source_sha256", "installed_config_sha256", "installed_opp_root", "runtime_opp_root", "runtime_opp_layout", "vendor", "vendor_impl_directory", "vendor_root",
                "source_file", "instrumentation", "hardware_envelope_heuristic", "strategy_algorithm_changes",
                "kernel_algorithm_changes", "formal_data_gate"):
        if item.get(key) != planned.get(key):
            raise RuntimeError("existing native GatherElements overlay provenance differs: {}".format(key))
    actual = Path(str(item["source_file"]))
    if not actual.is_file() or "__SOURCE_SHA256__" in actual.read_text(encoding="utf-8"):
        raise RuntimeError("native GatherElements overlay source is incomplete")
    content = actual.read_text(encoding="utf-8")
    for marker in ("GATHER_ELEMENTS_NATIVE_DYNAMIC_SOURCE_AUDIT_V1", CORE_ENV, UB_ENV, AUDIT_SCHEMA,
                   '@register_operator("{}")'.format(SOURCE_OPERATOR_TYPE)):
        if marker not in content:
            raise RuntimeError("native GatherElements overlay lost marker: {}".format(marker))
    if digest(actual) != str(item.get("source_file_sha256", "")):
        raise RuntimeError("native GatherElements overlay source hash changed")
    runtime_root = Path(str(item["runtime_opp_root"]))
    layout = item["runtime_opp_layout"]
    builtin = runtime_root / "built-in"
    priority = runtime_root / "vendors" / "config.ini"
    installed_root = Path(str(item["installed_opp_root"]))
    if (not runtime_root.is_dir() or not builtin.is_symlink() or
            builtin.resolve() != (installed_root / "built-in").resolve() or
            not priority.is_file() or priority.read_text(encoding="utf-8") != "load_priority={}\n".format(VENDOR) or
            Path(str(layout.get("vendor_priority_file", ""))) != priority):
        raise RuntimeError("native GatherElements private OPP layout is incomplete")
    return item


def prepare(cann: Path, output_parent: Path) -> dict[str, Any]:
    tbe, source, config = paths(cann)
    output = output_parent / "gather_elements_native_dynamic"
    original = source.read_text(encoding="utf-8")
    planned = expected(output, cann, source, config)
    prior = validate_existing(output, planned, original)
    if prior is not None:
        return prior
    if output.exists():
        raise RuntimeError("refuse to overwrite incomplete private native GatherElements overlay")
    root = Path(planned["runtime_opp_root"])
    vendor_root = Path(planned["vendor_root"])
    root.mkdir(parents=True)
    os.symlink(cann / "opp" / "built-in", root / "built-in", target_is_directory=True)
    os.symlink(cann / "opp" / "version.info", root / "version.info")
    priority = root / "vendors" / "config.ini"
    priority.parent.mkdir(parents=True)
    priority.write_text("load_priority={}\n".format(VENDOR), encoding="utf-8")
    target = Path(planned["source_file"])
    target.parent.mkdir(parents=True)
    text = instrumentation(original)
    # Embed an attested source variant hash after all source-preserving edits.
    pre_hash = hashlib.sha256(text.replace("__SOURCE_SHA256__", digest(source)).encode("utf-8")).hexdigest()
    target.write_text(text.replace("__SOURCE_SHA256__", pre_hash), encoding="utf-8")
    # The dynamic compiler imports CANN's normal ``impl.util`` helpers.  This
    # private link is read-only and leaves only the selected dynamic source in
    # the overlay itself.
    impl = vendor_root / "op_impl" / "ai_core" / "tbe" / str(planned["vendor_impl_directory"])
    os.symlink(tbe / "impl" / "util", impl / "util", target_is_directory=True)
    shutil.copy2(tbe / "impl" / "__init__.py", impl / "__init__.py")
    dynamic = impl / "dynamic"
    # ``target.parent`` above is this directory.  Reusing it is intentional:
    # the overlay contains exactly the patched GatherElements module plus the
    # installed dynamic package initializer.
    dynamic.mkdir(exist_ok=True)
    shutil.copy2(tbe / "impl" / "dynamic" / "__init__.py", dynamic / "__init__.py")
    config_data = json.loads(config.read_text(encoding="utf-8"))
    if "GatherElements" not in config_data:
        raise RuntimeError("installed CANN config has no GatherElements entry")
    custom_op_info = dict(config_data["GatherElements"])
    config_target = vendor_root / "op_impl" / "ai_core" / "tbe" / "config" / "ascend910b" / config.name
    config_target.parent.mkdir(parents=True)
    config_target.write_text(json.dumps({SOURCE_OPERATOR_TYPE: custom_op_info}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    planned["source_file_sha256"] = digest(target)
    planned["private_links"] = {
        "built_in": str(root / "built-in"),
        "vendor_impl_util": str(impl / "util"),
    }
    (output / "native_dynamic_overlay.json").write_text(json.dumps(planned, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return planned


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cann-root", required=True, type=Path)
    parser.add_argument("--output-parent", required=True, type=Path)
    args = parser.parse_args()
    if not args.output_parent.is_dir():
        raise RuntimeError("overlay parent is absent")
    print(json.dumps(prepare(args.cann_root, args.output_parent), sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as error:
        print("native GatherElements overlay error: {}".format(error), file=sys.stderr)
        raise SystemExit(2)
