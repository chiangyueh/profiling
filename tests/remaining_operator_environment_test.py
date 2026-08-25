#!/usr/bin/env python3
"""Static contract test for checkout-local remaining-operator workers."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source_adapter"))
import run_remaining_operator_campaign as remaining


def main() -> int:
    package = {
        "runtime_op": "gather_elements",
        "operator": "GatherElementsV2",
        "source_operator_type": "GatherElementsV2",
        "cann_root": "/private/cann81",
        "package_root": "/checkout/package/vendors/gather_elements_source",
        "op_api_library":
            "/checkout/package/vendors/gather_elements_source/op_api/lib/libcust_opapi.so",
        "instrumentation": {
            "opapi_library_environment": "GATHER_ELEMENTS_SOURCE_OPAPI_LIBRARY",
            "operator_type_environment": "GATHER_ELEMENTS_SOURCE_OPERATOR_TYPE",
            "audit_environment": "GATHER_ELEMENTS_TILING_AUDIT_PATH",
            "dispatch_environment": "GATHER_ELEMENTS_SOURCE_DISPATCH",
            "dispatch_value": "cann81_prebuilt_aclnn",
            "source_budget_environment": "GATHER_ELEMENTS_SOURCE_AIV_CAP",
        },
        "hardware_envelope_heuristic": {
            "environment": "GATHER_ELEMENTS_SOURCE_UB_DIVISOR",
        },
    }
    candidate = {"aiv_core_cap": 20, "hardware_envelope_divisor": 1}
    environment = remaining.source_environment(
        {"GATHER_ELEMENTS_SOURCE_OPERATOR_TYPE": "stale-value"},
        package, candidate, Path("/tmp/audit.jsonl"))
    expected = {
        "ASCEND_CUSTOM_OPP_PATH": package["package_root"],
        "GATHER_ELEMENTS_SOURCE_OPERATOR_TYPE": "GatherElementsV2",
        "GATHER_ELEMENTS_SOURCE_OPAPI_LIBRARY": package["op_api_library"],
        "GATHER_ELEMENTS_SOURCE_DISPATCH": "cann81_prebuilt_aclnn",
        "GATHER_ELEMENTS_SOURCE_AIV_CAP": "20",
        "GATHER_ELEMENTS_SOURCE_UB_DIVISOR": "1",
    }
    for name, value in expected.items():
        if environment.get(name) != value:
            raise AssertionError("{} expected {!r}, got {!r}".format(name, value, environment.get(name)))
    print("remaining_operator_environment_test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
