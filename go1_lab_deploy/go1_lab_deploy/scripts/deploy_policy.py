#!/usr/bin/env python3
"""HIMLoco deployment entry point; the default mode never opens LCM."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

# Support invocation from any working directory without installing training code.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path,
                        help="HIMLoco deployment JSON; model paths are relative to this file")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--check", action="store_true",
                       help="offline configuration/model preflight (default); no LCM")
    modes.add_argument("--inspect-config", action="store_true",
                       help="validate the configuration structure without loading a model")
    modes.add_argument("--observe", action="store_true",
                       help="receive robot state only; never publish motor commands")
    modes.add_argument("--run", action="store_true",
                       help="run the verified deployment; operator R2 transitions are still required")
    parser.add_argument("--duration", type=float, default=10.0,
                        help="state observation duration in seconds (only --observe)")
    parser.add_argument("--log", type=Path,
                        help="deployment JSONL log path (only --run)")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.duration <= 0 or not args.duration < float("inf"):
        build_parser().error("--duration must be finite and greater than zero")
    try:
        from go1_lab_deploy.himloco.config import load_config, readiness_errors

        config = load_config(args.config, require_ready=args.run)
        blockers = readiness_errors(config)
        ready = not blockers
        if args.inspect_config:
            print(json.dumps({"configuration_valid": True,
                              "deployment_ready": ready,
                              "readiness_blockers": blockers,
                              "model_loaded": False,
                              "motor_commands_published": False}, indent=2))
            return 0
        if args.observe:
            from go1_lab_deploy.himloco.runner import observe_states
            observe_states(config, duration=args.duration)
            return 0

        # Torch is imported lazily inside JitPolicy. Neither help nor config
        # inspection needs Torch, and no LCM object is created by preflight.
        import numpy as np
        from go1_lab_deploy.himloco.policy import JitPolicy

        model = JitPolicy(config["model"]["path"], config["model"]["sha256"])
        # JitPolicy performs its own five warmup calls before this preflight.
        action = np.asarray(model(np.zeros((1, 270), dtype=np.float32)))
        if action.shape != (12,) or not np.isfinite(action).all():
            raise ValueError("policy must produce twelve finite actions")
        if not args.run:
            print(json.dumps({"configuration_valid": True,
                              "model_loaded": True,
                              "model_sha256": config["model"]["sha256"],
                              "input_shape": [1, 270], "output_shape": [1, 12],
                              "deployment_ready": ready,
                              "readiness_blockers": blockers,
                              "motor_commands_published": False}, indent=2))
            return 0

        from go1_lab_deploy.himloco.runner import run_deployment
        log_path = args.log
        if log_path is None:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            log_path = PROJECT_ROOT / "logs" / ("himloco_" + stamp + ".jsonl")
        run_deployment(config, model, log_path=str(log_path))
        return 0
    except KeyboardInterrupt:
        # The runner owns command shutdown in its finally block.
        print("Interrupted.", file=sys.stderr)
        return 130
    except (ValueError, OSError, ImportError, RuntimeError, KeyError, TypeError) as exc:
        print("HIMLoco deployment error: {}".format(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
