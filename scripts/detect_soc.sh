#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/scripts/env.sh"
cd "$ROOT"

OUT_ENV="${1:-results/logs/detected_soc.env}"
DEVICE_ID="${DEVICE_ID:-0}"
mkdir -p "$(dirname "$OUT_ENV")" "$ROOT/build/probe"

SUPPORTED_SOC_REGEX='^Ascend910B(1|2|2C|3|4|4-1)$'

write_soc_env() {
    local soc="$1"
    local source="$2"
    local raw="$3"
    {
        echo "export ASCENDC_SOC_VERSION='${soc}'"
        echo "export SOC_VERSION='${soc}'"
        echo "export DETECTED_NPU_SOC='${soc}'"
        echo "export DETECTED_NPU_SOC_SOURCE='${source}'"
        echo "export DETECTED_NPU_SOC_RAW='${raw}'"
    } >"$OUT_ENV"
}

normalize_soc_from_text() {
    python3 -c '
import re
import sys

text = sys.stdin.read()
compact = re.sub(r"[\s_]+", "", text.upper())

exact = [
    ("Ascend910B4-1", ["ASCEND910B4-1", "910B4-1"]),
    ("Ascend910B2C", ["ASCEND910B2C", "910B2C"]),
    ("Ascend910B4", ["ASCEND910B4", "910B4"]),
    ("Ascend910B3", ["ASCEND910B3", "910B3"]),
    ("Ascend910B2", ["ASCEND910B2", "910B2"]),
    ("Ascend910B1", ["ASCEND910B1", "910B1"]),
]
for soc, patterns in exact:
    if any(pattern in compact for pattern in patterns):
        print(f"{soc}\texact")
        raise SystemExit(0)

if re.search(r"(^|[^0-9A-Z])(?:ASCEND)?910B([^0-9A-Z-]|$)", compact):
    print("Ascend910B\tgeneric_910B")
    raise SystemExit(0)

raise SystemExit(1)
'
}

extract_soc_or_empty() {
    local text="$1"
    local parsed
    parsed="$(printf '%s\n' "$text" | normalize_soc_from_text 2>/dev/null || true)"
    if [[ -n "$parsed" ]]; then
        echo "$parsed"
    fi
}

find_tool() {
    local name="$1"
    shift
    if command -v "$name" >/dev/null 2>&1; then
        command -v "$name"
        return 0
    fi
    local candidate
    for candidate in "$@"; do
        if [[ -x "$candidate" ]]; then
            echo "$candidate"
            return 0
        fi
    done
    return 1
}

driver_ld_path() {
    local paths=()
    local dir
    for dir in \
        /usr/local/Ascend/driver/lib64 \
        /usr/local/Ascend/driver/lib64/common \
        /usr/local/Ascend/driver/lib64/driver; do
        [[ -d "$dir" ]] && paths+=("$dir")
    done
    if [[ "${#paths[@]}" -gt 0 ]]; then
        (IFS=:; echo "${paths[*]}")
    fi
}

run_smi_once() {
    local smi_bin="$1"
    local ld_mode="$2"
    shift 2

    echo "\$ ${smi_bin} $*  (ld_mode=${ld_mode})"
    if [[ "$ld_mode" == "clean" ]]; then
        env -u LD_LIBRARY_PATH "$smi_bin" "$@"
    else
        local dlp
        dlp="$(driver_ld_path)"
        env LD_LIBRARY_PATH="${dlp}:${LD_LIBRARY_PATH:-}" "$smi_bin" "$@"
    fi
}

probe_with_npu_smi() {
    local smi_bin
    smi_bin="$(find_tool npu-smi /usr/local/Ascend/driver/tools/npu-smi || true)"
    if [[ -z "$smi_bin" ]]; then
        echo "npu-smi not found"
        return 20
    fi

    local output rc mode
    for mode in clean driver; do
        set +e
        output="$(run_smi_once "$smi_bin" "$mode" info 2>&1)"
        rc=$?
        set -e
        echo "$output"
        if [[ "$rc" -eq 0 ]]; then
            local extra
            extra="$(run_smi_once "$smi_bin" "$mode" info -t board 2>&1 || true)"
            output="${output}"$'\n'"${extra}"
            extra="$(run_smi_once "$smi_bin" "$mode" info -t product 2>&1 || true)"
            output="${output}"$'\n'"${extra}"
            local parsed
            parsed="$(extract_soc_or_empty "$output")"
            if [[ -n "$parsed" ]]; then
            local soc kind
            soc="${parsed%%$'\t'*}"
            kind="${parsed#*$'\t'}"
            if [[ "${kind}" != "exact" ]]; then
                echo "npu-smi exposed only a generic SoC (${soc}); exact Ascend910B submodel is required"
                return 24
            fi
            write_soc_env "$soc" "npu-smi:${kind}" "$soc"
            echo "detected_soc=${soc}"
            echo "source=npu-smi:${kind}"
            return 0
            fi
            echo "npu-smi ran successfully but did not expose an Ascend910B submodel"
            return 21
        fi

        if grep -Eaiq 'symbol lookup error|undefined symbol' <<<"$output"; then
            echo "npu-smi failed with a driver symbol error; this is treated as fatal"
            return 22
        fi
    done

    echo "npu-smi failed in all loader modes"
    return 23
}

probe_with_ascend_dmi() {
    local dmi_bin
    dmi_bin="$(find_tool ascend-dmi /usr/local/Ascend/driver/tools/ascend-dmi || true)"
    if [[ -z "$dmi_bin" ]]; then
        echo "ascend-dmi not found"
        return 30
    fi

    local output rc
    set +e
    output="$(env -u LD_LIBRARY_PATH "$dmi_bin" info 2>&1)"
    rc=$?
    set -e
    echo "$output"
    if [[ "$rc" -ne 0 ]]; then
        return 31
    fi

    local parsed
    parsed="$(extract_soc_or_empty "$output")"
    if [[ -z "$parsed" ]]; then
        return 32
    fi
    local soc kind
    soc="${parsed%%$'\t'*}"
    kind="${parsed#*$'\t'}"
    if [[ "${kind}" != "exact" ]]; then
        echo "ascend-dmi exposed only a generic SoC (${soc}); exact Ascend910B submodel is required"
        return 33
    fi
    write_soc_env "$soc" "ascend-dmi:${kind}" "$soc"
    echo "detected_soc=${soc}"
    echo "source=ascend-dmi:${kind}"
    return 0
}

probe_with_acl() {
    local src="$ROOT/build/probe/acl_soc_probe.cpp"
    local bin="$ROOT/build/probe/acl_soc_probe"
    cat >"$src" <<'CPP'
#include <acl/acl.h>
#include <acl/acl_base.h>
#include <acl/acl_rt.h>
#include <cstdlib>
#include <iostream>

int main() {
    const char *deviceEnv = std::getenv("DEVICE_ID");
    int deviceId = deviceEnv == nullptr ? 0 : std::atoi(deviceEnv);

    const aclError initRc = aclInit(nullptr);
    if (initRc != ACL_SUCCESS) {
        std::cerr << "aclInit failed, rc=" << initRc << std::endl;
        return 40;
    }

    const aclError setRc = aclrtSetDevice(deviceId);
    if (setRc != ACL_SUCCESS) {
        std::cerr << "aclrtSetDevice failed, rc=" << setRc << std::endl;
        aclFinalize();
        return 41;
    }

    const char *soc = aclrtGetSocName();
    if (soc == nullptr || soc[0] == '\0') {
        std::cerr << "aclrtGetSocName returned empty value" << std::endl;
        aclrtResetDevice(deviceId);
        aclFinalize();
        return 42;
    }
    std::cout << "aclrtGetSocName=" << soc << std::endl;

    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
CPP

    local include_dir="$CANN_PLATFORM_ROOT/include"
    local lib_dir="$CANN_PLATFORM_ROOT/lib64"
    local devlib_dir="$CANN_PLATFORM_ROOT/devlib"
    set +e
    g++ -std=c++17 -O2 "$src" -I"$include_dir" \
        -L"$lib_dir" -L"$devlib_dir" \
        -Wl,-rpath,"$lib_dir" -Wl,-rpath-link,"$devlib_dir" \
        -Wl,--allow-shlib-undefined \
        -lascendcl -o "$bin"
    local compile_rc=$?
    set -e
    if [[ "$compile_rc" -ne 0 ]]; then
        echo "acl SoC probe compile failed, rc=${compile_rc}"
        return 45
    fi

    local output rc
    set +e
    output="$(DEVICE_ID="$DEVICE_ID" "$bin" 2>&1)"
    rc=$?
    set -e
    echo "$output"
    if [[ "$rc" -ne 0 ]]; then
        return "$rc"
    fi

    local parsed
    parsed="$(extract_soc_or_empty "$output")"
    if [[ -z "$parsed" ]]; then
        return 43
    fi
    local soc kind
    soc="${parsed%%$'\t'*}"
    kind="${parsed#*$'\t'}"
    if [[ "${kind}" != "exact" ]]; then
        if [[ "${ACL_PROBE_ALLOW_GENERIC:-0}" == "1" && -n "${EXPECTED_EXACT_SOC:-}" ]]; then
            validate_detected_soc "$EXPECTED_EXACT_SOC"
            write_soc_env "$EXPECTED_EXACT_SOC" \
                "${EXPECTED_EXACT_SOC_SOURCE:-external}+aclrtGetSocName:${kind}" "$soc"
            echo "aclrtGetSocName exposed only a generic SoC (${soc}); keeping exact detector value ${EXPECTED_EXACT_SOC}"
            echo "detected_soc=${EXPECTED_EXACT_SOC}"
            echo "source=${EXPECTED_EXACT_SOC_SOURCE:-external}+aclrtGetSocName:${kind}"
            return 0
        fi
        echo "aclrtGetSocName exposed only a generic SoC (${soc}); exact Ascend910B submodel is required"
        return 44
    fi
    write_soc_env "$soc" "aclrtGetSocName:${kind}" "$soc"
    echo "detected_soc=${soc}"
    echo "source=aclrtGetSocName:${kind}"
    return 0
}

validate_detected_soc() {
    local soc="$1"
    if [[ ! "$soc" =~ $SUPPORTED_SOC_REGEX ]]; then
        echo "Unsupported detected SoC for this Ascend910B matmul package: ${soc}" >&2
        return 1
    fi
}

if [[ -n "${ASCENDC_SOC_VERSION:-}" && "${FORCE_ASCENDC_SOC_VERSION:-0}" == "1" ]]; then
    validate_detected_soc "$ASCENDC_SOC_VERSION"
    write_soc_env "$ASCENDC_SOC_VERSION" "env:ASCENDC_SOC_VERSION" "$ASCENDC_SOC_VERSION"
    echo "detected_soc=${ASCENDC_SOC_VERSION}"
    echo "source=env:ASCENDC_SOC_VERSION"
    exit 0
fi

echo "Detecting NPU SoC for device ${DEVICE_ID}"

if [[ -n "${DETECTED_ACL_SOC:-}" ]]; then
    parsed_acl_soc="$(extract_soc_or_empty "${DETECTED_ACL_SOC}")"
    if [[ -n "$parsed_acl_soc" ]]; then
        acl_soc="${parsed_acl_soc%%$'\t'*}"
        acl_kind="${parsed_acl_soc#*$'\t'}"
        if [[ "$acl_kind" == "exact" ]]; then
            validate_detected_soc "$acl_soc"
            write_soc_env "$acl_soc" "acl_runtime_probe:${acl_kind}" "$acl_soc"
            echo "detected_soc=${acl_soc}"
            echo "source=acl_runtime_probe:${acl_kind}"
            exit 0
        fi
    fi
fi

set +e
probe_with_npu_smi
smi_rc=$?
set -e
if [[ "$smi_rc" -eq 0 ]]; then
    # shellcheck disable=SC1090
    source "$OUT_ENV"
    smi_soc="$ASCENDC_SOC_VERSION"
    smi_source="$DETECTED_NPU_SOC_SOURCE"
    validate_detected_soc "$ASCENDC_SOC_VERSION"
    set +e
    EXPECTED_EXACT_SOC="$smi_soc" \
    EXPECTED_EXACT_SOC_SOURCE="$smi_source" \
    ACL_PROBE_ALLOW_GENERIC=1 \
        probe_with_acl
    acl_rc=$?
    set -e
    if [[ "$acl_rc" -ne 0 ]]; then
        echo "fatal: ACL runtime probe failed after npu-smi detected ${smi_soc}" >&2
        echo "  npu_smi_soc=${smi_soc}" >&2
        echo "  acl_probe_rc=${acl_rc}" >&2
        echo "  this run cannot reach NPU profiling until aclInit succeeds" >&2
        exit "$acl_rc"
    fi
    # shellcheck disable=SC1090
    source "$OUT_ENV"
    validate_detected_soc "$ASCENDC_SOC_VERSION"
    exit 0
fi

deferred_smi_error=0
if [[ "$smi_rc" =~ ^(22|23)$ && "${ALLOW_NPU_SMI_FAILURE:-0}" != "1" ]]; then
    deferred_smi_error=1
    echo "npu-smi probe failed; continuing only to look for an exact SoC from another source" >&2
fi

set +e
probe_with_ascend_dmi
dmi_rc=$?
set -e
if [[ "$dmi_rc" -eq 0 ]]; then
    # shellcheck disable=SC1090
    source "$OUT_ENV"
    dmi_soc="$ASCENDC_SOC_VERSION"
    dmi_source="$DETECTED_NPU_SOC_SOURCE"
    validate_detected_soc "$ASCENDC_SOC_VERSION"
    set +e
    EXPECTED_EXACT_SOC="$dmi_soc" \
    EXPECTED_EXACT_SOC_SOURCE="$dmi_source" \
    ACL_PROBE_ALLOW_GENERIC=1 \
        probe_with_acl
    acl_rc=$?
    set -e
    if [[ "$acl_rc" -ne 0 ]]; then
        echo "fatal: ACL runtime probe failed after ascend-dmi detected ${dmi_soc}" >&2
        echo "  ascend_dmi_soc=${dmi_soc}" >&2
        echo "  acl_probe_rc=${acl_rc}" >&2
        echo "  this run cannot reach NPU profiling until aclInit succeeds" >&2
        exit "$acl_rc"
    fi
    # shellcheck disable=SC1090
    source "$OUT_ENV"
    validate_detected_soc "$ASCENDC_SOC_VERSION"
    exit 0
fi

set +e
probe_with_acl
acl_rc=$?
set -e
if [[ "$acl_rc" -eq 0 ]]; then
    # shellcheck disable=SC1090
    source "$OUT_ENV"
    validate_detected_soc "$ASCENDC_SOC_VERSION"
    exit 0
fi

if [[ -n "${parsed_acl_soc:-}" && "${acl_kind:-}" == "exact" ]]; then
    validate_detected_soc "$acl_soc"
    write_soc_env "$acl_soc" "acl_runtime_probe:${acl_kind}" "$acl_soc"
    echo "detected_soc=${acl_soc}"
    echo "source=acl_runtime_probe:${acl_kind}"
    exit 0
fi

echo "fatal: unable to detect a supported Ascend910B SoC" >&2
echo "  npu_smi_rc=${smi_rc}" >&2
echo "  ascend_dmi_rc=${dmi_rc}" >&2
echo "  acl_probe_rc=${acl_rc}" >&2
if [[ "${deferred_smi_error}" == "1" ]]; then
    echo "  npu_smi_error=driver/tool query failed and no exact fallback was found" >&2
fi
if [[ -n "${acl_soc:-}" ]]; then
    echo "  acl_soc=${acl_soc}" >&2
    echo "  acl_soc_kind=${acl_kind:-unknown}" >&2
fi
echo "  output_env=${OUT_ENV}" >&2
exit 44
