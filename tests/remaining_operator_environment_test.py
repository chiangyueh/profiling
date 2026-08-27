#!/usr/bin/env python3
"""Static contract test for checkout-local remaining-operator workers."""

from pathlib import Path
import inspect
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source_adapter"))
import run_remaining_operator_campaign as remaining
import prepare_fasg_strategy_overlays as fasg_overlay
import prepare_fias_source_overlay as fias_overlay


def main() -> int:
    public_entry = (ROOT / "run_npu.sh").read_text(encoding="utf-8")
    remaining_entry = (ROOT / "run_remaining_npu.sh").read_text(encoding="utf-8")
    if "--operator all" not in public_entry:
        raise AssertionError("full entry does not dispatch the remaining-operator campaign")
    expected_loop = "for selected in flash_attention_score_grad fused_infer_attention_score; do"
    if expected_loop not in remaining_entry:
        raise AssertionError("full campaign is not restricted to FASG and FIAS")
    if "for selected in gather_elements" in remaining_entry:
        raise AssertionError("completed GatherElements is still scheduled by full mode")
    if "--target package" in remaining_entry or "--target ops_kernel" in remaining_entry:
        raise AssertionError("remaining campaign still triggers device-kernel compilation")
    if "fasg_official_semantic_dispatch" not in remaining_entry:
        raise AssertionError("FASG does not build the complete official dispatcher")
    for forbidden in ("FASG_PROJECTS", "materialize_attention_package_variant.py",
                      "fasg_flashattentionscoregradtilings1s2bn2gs1s2"):
        if forbidden in remaining_entry:
            raise AssertionError("obsolete isolated FASG route is still active: " + forbidden)
    required_host_targets = (
        "opapi opsproto optiling optiling_compat generate_ops_info_ascend910b",
        "materialize_installed_attention_kernels.py",
    )
    if any(value not in remaining_entry for value in required_host_targets):
        raise AssertionError("host-only package or installed-kernel materialization is incomplete")

    seed = """    compileInfoPtr->aivNum = ascendcPlatform.GetCoreNumAiv();
    compileInfoPtr->aicNum = ascendcPlatform.GetCoreNumAic();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, compileInfoPtr->l2CacheSize);"""
    instrumented = fasg_overlay.instrument_source_core_budget(seed)
    if instrumented.index("GetCoreMemSize(platform_ascendc::CoreMemType::L2") > instrumented.index(
            "const char *l2DivisorText"):
        raise AssertionError("FASG L2 divisor is applied before the physical capacity query")
    for function in (fasg_overlay.instrument_source_core_budget,
                     fias_overlay.instrument_fias_tiler, fias_overlay.instrument_ifa_tiler):
        implementation = inspect.getsource(function)
        if "runtimeAiv % runtimeAic" not in implementation or "runtimeAic % runtimeAiv" in implementation:
            raise AssertionError("attention core-cap instrumentation does not preserve the 910B AIV/AIC ratio")

    dispatcher_source = inspect.getsource(fasg_overlay.write_dispatcher_overlay)
    for required in ("original_strategy_registry_preserved", "enabled_original_registrations",
                     '"disabled_original_registrations": []'):
        if required not in dispatcher_source:
            raise AssertionError("FASG dispatcher attestation is incomplete: " + required)
    valid_fasg = {
        "runtime_op": "flash_attention_score_grad",
        "strategy_class": "official_semantic_dispatch",
        "original_strategy_registry_preserved": True,
        "enabled_original_registrations": [
            {"class": "Strategy{}".format(index), "priority": index} for index in range(8)],
        "disabled_original_registrations": [],
    }
    if remaining.plan_packages([valid_fasg], "flash_attention_score_grad") != {
            "flash_attention_score_grad": [valid_fasg]}:
        raise AssertionError("complete FASG dispatcher package was rejected")
    invalid_fasg = dict(valid_fasg)
    invalid_fasg["strategy_class"] = "FlashAttentionScoreGradTilingS1s2Bn2gs1s2"
    try:
        remaining.plan_packages([invalid_fasg], "flash_attention_score_grad")
    except RuntimeError:
        pass
    else:
        raise AssertionError("isolated FASG strategy package was accepted")

    for short, operator, prefix, envelope in (
        ("flash_attention_score_grad", "FlashAttentionScoreGrad", "FASG", "L2"),
        ("fused_infer_attention_score", "FusedInferAttentionScore", "FIAS", "UB"),
    ):
        package = {
            "runtime_op": short,
            "operator": operator,
            "cann_root": "/private/cann81",
            "package_root": "/checkout/package/vendors/attention_source",
            "op_api_library": "/checkout/package/vendors/attention_source/op_api/lib/libcust_opapi.so",
            "instrumentation": {
                "opapi_library_environment": prefix + "_SOURCE_OPAPI_LIBRARY",
                "audit_environment": prefix + "_TILING_AUDIT_PATH",
                "dispatch_environment": prefix + "_SOURCE_DISPATCH",
                "dispatch_value": "cann81_prebuilt_aclnn",
                "source_budget_environment": prefix + "_SOURCE_AIV_CAP",
            },
            "hardware_envelope_heuristic": {
                "environment": prefix + "_SOURCE_" + envelope + "_DIVISOR",
            },
        }
        candidate = {"aiv_core_cap": 20, "hardware_envelope_divisor": 1}
        environment = remaining.source_environment({}, package, candidate, Path("/tmp/audit.jsonl"))
        expected = {
            "ASCEND_CUSTOM_OPP_PATH": package["package_root"],
            prefix + "_SOURCE_OPAPI_LIBRARY": package["op_api_library"],
            prefix + "_SOURCE_DISPATCH": "cann81_prebuilt_aclnn",
            prefix + "_SOURCE_AIV_CAP": "20",
            prefix + "_SOURCE_" + envelope + "_DIVISOR": "1",
        }
        for name, value in expected.items():
            if environment.get(name) != value:
                raise AssertionError("{} expected {!r}, got {!r}".format(name, value, environment.get(name)))
    print("remaining_operator_environment_test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
