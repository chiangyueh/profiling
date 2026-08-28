#!/usr/bin/env bash
# CANN environment for this project.
#
# Prefer the official CANN set_env.sh. Hand-built LD_LIBRARY_PATH lists are easy
# to get wrong on NPU hosts because toolkit, runtime and driver libraries must
# stay ABI-compatible.

set -euo pipefail

export CANN_ROOT="${CANN_ROOT:-/usr/local/Ascend/ascend-toolkit/latest}"
export SOC_VERSION="${SOC_VERSION:-Ascend910B}"
export CANN_ARCH="${CANN_ARCH:-$(uname -m)}"

filter_ascend_path() {
    local value="${1:-}"
    python3 - "$value" <<'PY'
import os
import sys

raw = sys.argv[1] if len(sys.argv) > 1 else ""
drop_tokens = (
    "/usr/local/Ascend/ascend-toolkit",
    "/usr/local/Ascend/driver",
    "/usr/local/Ascend/nnrt",
)
kept = []
seen = set()
for item in raw.split(":"):
    if not item:
        continue
    if any(token in item for token in drop_tokens):
        continue
    if item in seen:
        continue
    seen.add(item)
    kept.append(item)
print(":".join(kept))
PY
}

prepend_path_unique() {
    local var_name="$1"
    local entry="$2"
    [[ -d "${entry}" ]] || return 0
    local current="${!var_name:-}"
    case ":${current}:" in
        *":${entry}:"*) ;;
        *) export "${var_name}=${entry}${current:+:${current}}" ;;
    esac
}

append_path_unique() {
    local var_name="$1"
    local entry="$2"
    [[ -d "${entry}" ]] || return 0
    local current="${!var_name:-}"
    case ":${current}:" in
        *":${entry}:"*) ;;
        *) export "${var_name}=${current:+${current}:}${entry}" ;;
    esac
}

prepend_paths_in_order() {
    local var_name="$1"
    shift
    local i
    for ((i = $#; i >= 1; --i)); do
        prepend_path_unique "${var_name}" "${!i}"
    done
}

find_set_env() {
    local root="$1"
    local parent
    parent="$(dirname "$root")"
    for candidate in \
        "${root}/set_env.sh" \
        "${parent}/set_env.sh" \
        "/usr/local/Ascend/ascend-toolkit/set_env.sh"; do
        if [[ -f "${candidate}" ]]; then
            echo "${candidate}"
            return 0
        fi
    done
    return 1
}

USED_OFFICIAL_SET_ENV=0
if [[ "${ASCEND_MATMUL_MANUAL_ENV:-0}" != "1" ]]; then
    if [[ "${ASCEND_MATMUL_STRICT_ENV:-1}" == "1" ]]; then
        export LD_LIBRARY_PATH="$(filter_ascend_path "${LD_LIBRARY_PATH:-}")"
        export PYTHONPATH="$(filter_ascend_path "${PYTHONPATH:-}")"
        export PATH="$(filter_ascend_path "${PATH:-}")"
        unset ASCEND_HOME_PATH ASCEND_TOOLKIT_HOME ASCEND_AICPU_PATH ASCEND_OPP_PATH TOOLCHAIN_HOME || true
    fi
    SET_ENV_SH="$(find_set_env "${CANN_ROOT}" || true)"
    if [[ -n "${SET_ENV_SH}" ]]; then
        : "${LD_LIBRARY_PATH:=}"
        : "${PYTHONPATH:=}"
        set +u
        # shellcheck disable=SC1090
        source "${SET_ENV_SH}"
        set -u
        export CANN_ROOT="${ASCEND_TOOLKIT_HOME:-${CANN_ROOT}}"
        USED_OFFICIAL_SET_ENV=1
    fi
    unset SET_ENV_SH
fi

export CANN_PLATFORM_ROOT="${CANN_ROOT}/${CANN_ARCH}-linux"

if [[ ! -d "${CANN_ROOT}" ]]; then
    echo "CANN root not found: ${CANN_ROOT}" >&2
    exit 1
fi

if [[ ! -d "${CANN_PLATFORM_ROOT}" ]]; then
    echo "CANN platform directory not found: ${CANN_PLATFORM_ROOT}" >&2
    echo "Detected arch: ${CANN_ARCH}" >&2
    exit 1
fi

export ASCEND_TOOLKIT_HOME="${CANN_ROOT}"
export ASCEND_HOME_PATH="${CANN_ROOT}"
export ASCEND_AICPU_PATH="${CANN_ROOT}"
export ASCEND_LATEST_INSTALL_PATH="${ASCEND_LATEST_INSTALL_PATH:-/usr/local/Ascend}"
[[ -d "${CANN_ROOT}/opp" ]] && export ASCEND_OPP_PATH="${CANN_ROOT}/opp"
[[ -d "${CANN_ROOT}/toolkit" ]] && export TOOLCHAIN_HOME="${CANN_ROOT}/toolkit"

_runtime_ld_paths=()
[[ -d "${CANN_PLATFORM_ROOT}/lib64" ]] && _runtime_ld_paths+=("${CANN_PLATFORM_ROOT}/lib64")
[[ -d "${CANN_ROOT}/lib64" ]] && _runtime_ld_paths+=("${CANN_ROOT}/lib64")
[[ -d "${CANN_ROOT}/runtime/lib64" ]] && _runtime_ld_paths+=("${CANN_ROOT}/runtime/lib64")
[[ -d "${CANN_ROOT}/fwkacllib/lib64" ]] && _runtime_ld_paths+=("${CANN_ROOT}/fwkacllib/lib64")
[[ -d "${CANN_ROOT}/atc/lib64" ]] && _runtime_ld_paths+=("${CANN_ROOT}/atc/lib64")
[[ -d /usr/local/Ascend/driver/lib64 ]] && _runtime_ld_paths+=("/usr/local/Ascend/driver/lib64")
[[ -d /usr/local/Ascend/driver/lib64/common ]] && _runtime_ld_paths+=("/usr/local/Ascend/driver/lib64/common")
[[ -d /usr/local/Ascend/driver/lib64/driver ]] && _runtime_ld_paths+=("/usr/local/Ascend/driver/lib64/driver")
if [[ "${#_runtime_ld_paths[@]}" -gt 0 ]]; then
    prepend_paths_in_order LD_LIBRARY_PATH "${_runtime_ld_paths[@]}"
fi
unset _runtime_ld_paths

# devlib contains toolkit-side compile/link stubs on many installations. Keep it
# as a final fallback only; real driver libraries above must win at runtime.
append_path_unique LD_LIBRARY_PATH "${CANN_PLATFORM_ROOT}/devlib"

if [[ "${USED_OFFICIAL_SET_ENV}" == "1" ]]; then
    unset USED_OFFICIAL_SET_ENV
    return 0 2>/dev/null || exit 0
fi
unset USED_OFFICIAL_SET_ENV

_cann_ld_paths=()
[[ -d "${CANN_PLATFORM_ROOT}/lib64" ]] && _cann_ld_paths+=("${CANN_PLATFORM_ROOT}/lib64")
[[ -d "${CANN_ROOT}/lib64" ]] && _cann_ld_paths+=("${CANN_ROOT}/lib64")
[[ -d "${CANN_ROOT}/runtime/lib64" ]] && _cann_ld_paths+=("${CANN_ROOT}/runtime/lib64")
[[ -d "${CANN_ROOT}/fwkacllib/lib64" ]] && _cann_ld_paths+=("${CANN_ROOT}/fwkacllib/lib64")
[[ -d "${CANN_ROOT}/atc/lib64" ]] && _cann_ld_paths+=("${CANN_ROOT}/atc/lib64")
[[ -d /usr/local/Ascend/driver/lib64 ]] && _cann_ld_paths+=("/usr/local/Ascend/driver/lib64")
[[ -d /usr/local/Ascend/driver/lib64/common ]] && _cann_ld_paths+=("/usr/local/Ascend/driver/lib64/common")
[[ -d /usr/local/Ascend/driver/lib64/driver ]] && _cann_ld_paths+=("/usr/local/Ascend/driver/lib64/driver")
[[ -d "${CANN_ROOT}/tools/aml/lib64" ]] && _cann_ld_paths+=("${CANN_ROOT}/tools/aml/lib64")
[[ -d "${CANN_ROOT}/tools/aml/lib64/plugin" ]] && _cann_ld_paths+=("${CANN_ROOT}/tools/aml/lib64/plugin")
[[ -d "${CANN_ROOT}/opp/built-in/op_impl/ai_core/tbe/op_tiling/lib/linux/${CANN_ARCH}" ]] && \
    _cann_ld_paths+=("${CANN_ROOT}/opp/built-in/op_impl/ai_core/tbe/op_tiling/lib/linux/${CANN_ARCH}")

if [[ "${#_cann_ld_paths[@]}" -gt 0 ]]; then
    _cann_ld_path="$(IFS=:; echo "${_cann_ld_paths[*]}")"
    export LD_LIBRARY_PATH="${_cann_ld_path}:${LD_LIBRARY_PATH:-}"
    unset _cann_ld_path
fi
unset _cann_ld_paths

_cann_path_entries=()
[[ -d "${CANN_ROOT}/bin" ]] && _cann_path_entries+=("${CANN_ROOT}/bin")
[[ -d "${CANN_ROOT}/compiler/ccec_compiler/bin" ]] && _cann_path_entries+=("${CANN_ROOT}/compiler/ccec_compiler/bin")
[[ -d "${CANN_ROOT}/tools/ccec_compiler/bin" ]] && _cann_path_entries+=("${CANN_ROOT}/tools/ccec_compiler/bin")
[[ -d "${CANN_ROOT}/compiler/bishengir/bin" ]] && _cann_path_entries+=("${CANN_ROOT}/compiler/bishengir/bin")
if [[ "${#_cann_path_entries[@]}" -gt 0 ]]; then
    _cann_path="$(IFS=:; echo "${_cann_path_entries[*]}")"
    export PATH="${_cann_path}:${PATH:-}"
    unset _cann_path
fi
unset _cann_path_entries

_cann_python_paths=()
[[ -d "${CANN_ROOT}/python/site-packages" ]] && _cann_python_paths+=("${CANN_ROOT}/python/site-packages")
[[ -d "${CANN_ROOT}/opp/built-in/op_impl/ai_core/tbe" ]] && _cann_python_paths+=("${CANN_ROOT}/opp/built-in/op_impl/ai_core/tbe")
if [[ "${#_cann_python_paths[@]}" -gt 0 ]]; then
    _cann_python_path="$(IFS=:; echo "${_cann_python_paths[*]}")"
    export PYTHONPATH="${_cann_python_path}:${PYTHONPATH:-}"
    unset _cann_python_path
fi
unset _cann_python_paths
