"""Compare A/A repeatability and A/B instrumentation on bounded state traces.

This checks the recorded prefix, not all future trajectories or force accuracy.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def compare(reference, candidate, atol=1e-5):
    roots = [Path(reference), Path(candidate)]
    reports = [json.loads((root / "report.json").read_text()) for root in roots]
    for key in ("status", "num_envs", "steps", "scenario", "seed", "velocity", "randomization",
                "observation_noise", "forced_timeout_step", "window", "policy_sha256"):
        if reports[0][key] != reports[1][key] or reports[0]["status"] != "completed":
            raise ValueError(f"Incomparable collection field: {key}")
    errors = {}
    with np.load(roots[0] / "behavior_trace.npz", allow_pickle=False) as a, np.load(
            roots[1] / "behavior_trace.npz", allow_pickle=False) as b:
        if set(a.files) != set(b.files):
            raise ValueError("Behavior schemas differ")
        for key in a.files:
            if a[key].shape != b[key].shape or not np.isfinite(a[key]).all() or not np.isfinite(b[key]).all():
                raise ValueError(f"Invalid comparison array: {key}")
            error = float(np.max(np.abs(a[key].astype(float) - b[key].astype(float))))
            errors[key] = error
    return {"passed": all(error <= atol for error in errors.values()), "atol": atol,
            "max_abs_error": errors, "reference": str(roots[0]), "candidate": str(roots[1]),
            "scope": "up to eight recorded environments; finite prefix; run A/A before interpreting A/B"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference")
    parser.add_argument("candidate")
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not np.isfinite(args.atol) or args.atol < 0:
        parser.error("atol must be finite and nonnegative")
    result = compare(args.reference, args.candidate, args.atol)
    if args.output:
        with args.output.open("x") as stream:
            stream.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)
