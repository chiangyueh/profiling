#!/usr/bin/env bash
# Read-only CANN TBE Python import check for GatherElements source dispatch.
# This script does not call ACL, touch an NPU, compile, or write files.
set -Eeuo pipefail

CANN_ROOT="${CANN_ROOT:-/usr/local/Ascend/ascend-toolkit/latest}"
[[ -d "${CANN_ROOT}" && -f "${CANN_ROOT}/opp/version.info" ]] || {
    echo "GATHER_TBE_PYTHON_PROBE {\"status\":\"failed\",\"error\":\"CANN root or OPP package is missing\"}"
    exit 1
}

ENV_FILE=""
for candidate in "${CANN_ROOT}/set_env.sh" "$(dirname "${CANN_ROOT}")/set_env.sh"; do
    [[ -f "${candidate}" ]] && { ENV_FILE="${candidate}"; break; }
done
[[ -n "${ENV_FILE}" ]] || {
    echo "GATHER_TBE_PYTHON_PROBE {\"status\":\"failed\",\"error\":\"CANN environment script is missing\"}"
    exit 1
}

# This script is run with `bash`, so sourcing changes only this child process.
set +u
source "${ENV_FILE}"
set -u
export PYTHONDONTWRITEBYTECODE=1

python3 - <<'PY'
import importlib
import json
import sys

result = {"status": "passed", "python": sys.executable,
          "version": sys.version.split()[0], "modules": {}}
for name in ("te", "tbe"):
    try:
        module = importlib.import_module(name)
        result["modules"][name] = {"status": "ok", "file": getattr(module, "__file__", None)}
    except Exception as error:
        result["modules"][name] = {"status": "failed", "error": repr(error)}
        result["status"] = "failed"
print("GATHER_TBE_PYTHON_PROBE " + json.dumps(result, sort_keys=True))
raise SystemExit(0 if result["status"] == "passed" else 1)
PY
