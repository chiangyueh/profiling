#!/usr/bin/env bash
set -euo pipefail

cann_root="${CANN_ROOT:-${ASCEND_TOOLKIT_HOME:-/usr/local/Ascend/ascend-toolkit/latest}}"
real_linker="${cann_root}/tools/ccec_compiler/bin/ld.lld"
if [[ ! -x "${real_linker}" ]]; then
    real_linker="${cann_root}/compiler/ccec_compiler/bin/ld.lld"
fi
if [[ ! -x "${real_linker}" ]]; then
    echo "fatal: CCEC ld.lld was not found under ${cann_root}" >&2
    exit 127
fi

exec "${real_linker}" --threads=1 "$@"
