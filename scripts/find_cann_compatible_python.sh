#!/usr/bin/env bash
# Read-only selector probe for a CANN-8.1-compatible Python executable.
# It does not call ACL, touch an NPU, compile, or write files.
set -Eeuo pipefail

CANN_ROOT="${CANN_ROOT:-/usr/local/Ascend/ascend-toolkit/latest}"
ENV_FILE=""
for candidate in "${CANN_ROOT}/set_env.sh" "$(dirname "${CANN_ROOT}")/set_env.sh"; do
    [[ -f "${candidate}" ]] && { ENV_FILE="${candidate}"; break; }
done
[[ -n "${ENV_FILE}" ]] || {
    echo "CANN_PYTHON_SELECTOR {\"status\":\"failed\",\"error\":\"CANN environment script is missing\"}"
    exit 1
}

set +u
source "${ENV_FILE}"
set -u
export PYTHONDONTWRITEBYTECODE=1

declare -A seen=()
candidates=()
for name in /usr/bin/python3 /usr/bin/python3.7 /usr/bin/python3.8 /usr/bin/python3.9 /usr/bin/python3.10 /usr/bin/python3.11 \
            python3.7 python3.8 python3.9 python3.10 python3.11 python3; do
    if [[ "${name}" == /* ]]; then
        [[ -x "${name}" ]] || continue
        path="${name}"
    else
        path="$(command -v "${name}" 2>/dev/null || true)"
        [[ -n "${path}" && -x "${path}" ]] || continue
    fi
    real="$(readlink -f "${path}")"
    [[ -n "${seen[${real}]:-}" ]] && continue
    seen["${real}"]=1
    candidates+=("${real}")
done

[[ ${#candidates[@]} -gt 0 ]] || {
    echo "CANN_PYTHON_SELECTOR {\"status\":\"failed\",\"error\":\"no Python executable found\"}"
    exit 1
}

python3 - "${candidates[@]}" <<'PY'
import json
import os
import subprocess
import sys

rows = []
probe = r'''
import importlib, json, sys
row = {"path": sys.executable, "version": list(sys.version_info[:3])}
row["cann_81_supported"] = (3, 7, 0) <= tuple(sys.version_info[:3]) <= (3, 11, 4)
try:
    te = importlib.import_module("te")
    tbe = importlib.import_module("tbe")
    row["imports"] = {"status": "ok", "te": getattr(te, "__file__", None), "tbe": getattr(tbe, "__file__", None)}
except Exception as error:
    row["imports"] = {"status": "failed", "error": repr(error)}
print(json.dumps(row, sort_keys=True))
'''
for executable in sys.argv[1:]:
    completed = subprocess.run([executable, "-c", probe], text=True, capture_output=True,
                               env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"), check=False)
    try:
        row = json.loads(completed.stdout.strip())
    except json.JSONDecodeError:
        row = {"path": executable, "imports": {"status": "failed", "error": completed.stderr.strip()[-400:]}}
    row["selected"] = bool(row.get("cann_81_supported") and row.get("imports", {}).get("status") == "ok")
    rows.append(row)
selected = [row["path"] for row in rows if row["selected"]]
print("CANN_PYTHON_SELECTOR " + json.dumps({"status": "passed", "selected": selected, "candidates": rows}, sort_keys=True))
PY
