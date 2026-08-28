#!/usr/bin/env python3
import argparse
import json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-m", type=int, required=True)
    parser.add_argument("--base-n", type=int, required=True)
    parser.add_argument("--single-m", type=int, required=True)
    parser.add_argument("--single-n", type=int, required=True)
    parser.add_argument("--step-m", type=int, required=True)
    parser.add_argument("--step-n", type=int, required=True)
    args = parser.parse_args()

    rules = []
    if args.single_m > args.step_m * args.base_m:
        rules.append("BASE_SINGLE_CORE_M_EXCEEDS_STEP_M_BASE_M")
    if args.single_n > args.step_n * args.base_n:
        rules.append("BASE_SINGLE_CORE_N_EXCEEDS_STEP_N_BASE_N")

    print(json.dumps({
        "validator": {
            "status": "rejected" if rules else "passed",
            "rules": rules,
        }
    }, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
