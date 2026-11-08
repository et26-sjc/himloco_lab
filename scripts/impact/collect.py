"""A0: frozen Go1 contact collection. No PPO or impact-model training.

Writes time-major NPZ shards; raw frames allow exact six-frame reconstruction
without storing six duplicate copies of adjacent measurements. Launch from repo
root in conda himloco_lab. --help does not start SimulationApp.
"""
import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

from isaaclab.app import AppLauncher


REPO = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--plan", type=Path, default=REPO / "configs/impact/go1_impact_replay.json")
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--num_envs", type=int)
parser.add_argument("--steps", type=int)
parser.add_argument("--shard-steps", type=int)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--scenario", choices=("plane_walk", "rough_walk", "suite_walk", "stand", "drop", "airborne"))
parser.add_argument("--physics-dt", type=float)
parser.add_argument("--solver-position-iterations", type=int)
parser.add_argument("--solver-velocity-iterations", type=int)
parser.add_argument("--research-observations", action="store_true",
                    help="Archive clean 200Hz proprioception, ideal IMU specific force and actual actuator effort")
parser.add_argument("--him-supervision", action="store_true",
                    help="Archive next critic first48 only for original HIM losses, with pre-reset terminal replacement")
parser.add_argument("--drop-height-range", type=float, nargs=2,
                    help="Controlled randomized initial base height, only for drop diagnostics [m]")
parser.add_argument("--velocity", type=float, nargs=3)
parser.add_argument("--velocity-range-x", type=float, nargs=2,
                    help="Research override: uniform forward command range, resampled only on reset")
parser.add_argument("--episode-length-s", type=float,
                    help="Research override for safe travel inside the fixed terrain tiles")
parser.add_argument("--randomization", action=argparse.BooleanOptionalAction, default=None)
parser.add_argument("--observation-noise", action=argparse.BooleanOptionalAction, default=None)
parser.add_argument("--baseline-observer", action="store_true", help="No added sensor; for A/A and A/B trajectory audits")
parser.add_argument("--audit-baseline-contracts", action="store_true", help="Run existing HIM/PD contract tests after AppLauncher")
parser.add_argument("--force-timeout-step", type=int, help="Diagnostic override only: force every row to timeout at this zero-based step")
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--headless", action="store_true")
args = parser.parse_args()
plan = json.loads(args.plan.read_text())
collection = plan["collection"]
for key in ("num_envs", "steps", "shard_steps", "scenario", "physics_dt", "velocity", "randomization", "observation_noise"):
    if getattr(args, key) is None:
        setattr(args, key, collection[key])
if min(args.num_envs, args.steps, args.shard_steps) < 1:
    parser.error("environment, step and shard counts must be positive")
if args.force_timeout_step is not None and not 0 <= args.force_timeout_step < args.steps:
    parser.error("forced timeout step must lie inside this collection")
# The raw collector is threshold-agnostic. Derived labels and their immutable
# version live in the downstream experiment spec, not in simulator stepping.
args.output = args.output.resolve()
args.output.mkdir(parents=True, exist_ok=False)
(args.output / "INCOMPLETE").write_text("Collection is not valid until report.json says completed.\n")
launcher = AppLauncher(args)
sys.argv = [sys.argv[0]]

import numpy as np
import torch

from himloco_lab.envs import HimlocoManagerBasedRLEnv
from himloco_lab.impact.collection_cfg import make_collection_cfg
from himloco_lab.impact.labels import ContactWindow, completed_interval_frame, contact_window_targets
from himloco_lab.impact.recording_env import FOOT_NAMES, ImpactRecordingEnv
from himloco_lab.rsl_rl import HimlocoVecEnvWrapper
from himloco_lab.rsl_rl.wrappers.history import stack_frame
from isaaclab.utils.io import dump_yaml


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def array(value):
    return value.detach().cpu().numpy().copy()


def main():
    torch.set_num_threads(4)
    baseline_contract_tests = 0
    if args.audit_baseline_contracts:
        import unittest
        suite = unittest.TestSuite()
        for pattern in ("test_him_contracts.py", "test_pd_baseline.py"):
            suite.addTests(unittest.defaultTestLoader.discover(str(REPO / "tests"), pattern=pattern))
        baseline_contract_tests = suite.countTestCases()
        if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():
            raise AssertionError("Existing HIM/PD baseline contract tests failed")
    if not np.isfinite(args.velocity).all():
        raise ValueError("Non-finite velocity")
    baseline = plan["baseline"]
    policy_path = REPO / baseline["policy"]
    checkpoint_path = REPO / baseline["checkpoint"]
    for path, expected in ((policy_path, baseline["policy_sha256"]),
                           (checkpoint_path, baseline["checkpoint_sha256"])):
        if sha256(path) != expected:
            raise ValueError(f"Baseline hash mismatch: {path}")
    window = ContactWindow(args.physics_dt)
    if window.policy_dt != collection["policy_dt"] or window.rise_interval_s != collection["rise_interval_s"]:
        raise ValueError("Window constants and plan disagree")
    cfg = make_collection_cfg(args.num_envs, args.seed, args.scenario, args.physics_dt, tuple(args.velocity),
                              randomization=args.randomization, observation_noise=args.observation_noise,
                              instrumented=not args.baseline_observer)
    cfg.sim.device = args.device
    if args.velocity_range_x is not None:
        low, high = args.velocity_range_x
        if not (0 < low <= high <= 1.2):
            raise ValueError('Forward command range must lie in (0,1.2]')
        cfg.commands.base_velocity.ranges.lin_vel_x = (low, high)
        cfg.commands.base_velocity.low_vel_env_lin_x_ranges = (low, high)
    if args.episode_length_s is not None:
        if not (1 <= args.episode_length_s <= 6):
            raise ValueError('Research episode length must lie in [1,6] seconds')
        cfg.episode_length_s = args.episode_length_s
    if args.solver_position_iterations is not None:
        cfg.scene.robot.spawn.articulation_props.solver_position_iteration_count = args.solver_position_iterations
        cfg.sim.physics.min_position_iteration_count = args.solver_position_iterations
        cfg.sim.physics.max_position_iteration_count = args.solver_position_iterations
    if args.solver_velocity_iterations is not None:
        cfg.scene.robot.spawn.articulation_props.solver_velocity_iteration_count = args.solver_velocity_iterations
        cfg.sim.physics.min_velocity_iteration_count = args.solver_velocity_iterations
        cfg.sim.physics.max_velocity_iteration_count = args.solver_velocity_iterations
    if args.drop_height_range is not None:
        low, high = args.drop_height_range
        if args.scenario != 'drop' or not (0.35 <= low <= high <= 1.0):
            raise ValueError('Drop height range must be ordered in [.35,1.0] and use drop scenario')
        cfg.events.reset_base.params['pose_range']['z'] = (low - 0.6, high - 0.6)
    cfg.log_dir = str(args.output)
    dump_yaml(str(args.output / "env.yaml"), cfg)
    (args.output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    source_paths = [Path(__file__), args.plan.resolve()]
    source_paths += sorted((REPO / "source/himloco_lab/himloco_lab/impact").glob("*.py"))
    before_policy_hash = sha256(policy_path)
    walk = args.scenario.endswith("walk")
    policy = torch.jit.load(str(policy_path), map_location=args.device).eval() if walk else None
    if policy is not None:
        for param in policy.parameters():
            param.requires_grad_(False)
    start = time.perf_counter()
    raw_env = (HimlocoManagerBasedRLEnv if args.baseline_observer else ImpactRecordingEnv)(cfg=cfg)
    env = HimlocoVecEnvWrapper(raw_env, history_length=5, privileged_history_length=0)
    startup_s = time.perf_counter() - start
    try:
        if (env.num_one_step_obs, env.num_obs, env.num_privileged_obs, env.num_actions) != (45, 270, 238, 12):
            raise AssertionError("Unexpected baseline interface")
        if type(raw_env.scene["robot"].actuators["base_legs"]).__name__ != "IdealPDActuator":
            raise AssertionError("Expected explicit baseline PD")
        episode = torch.zeros(args.num_envs, dtype=torch.int64, device=env.device)
        age = torch.zeros_like(episode)
        reset_ids = torch.arange(args.num_envs, device=env.device)
        reconstructed = torch.zeros(args.num_envs, 270, device=env.device)
        pending = []
        shards = []
        summary = []
        reset_count = 0
        captured_terminals = 0
        history_error = 0.0
        trace = []
        robot = raw_env.scene["robot"]
        observer = None
        if args.research_observations:
            from himloco_lab.impact.physics_observer import PhysicsObserver
            observer = PhysicsObserver(raw_env)
        initial_root = array(robot.data.root_state_w.torch)
        np.savez(args.output / 'initial_root_state.npz', root_state_w=initial_root)
        terrain_metadata = None
        if args.scenario == "suite_walk":
            from himloco_lab.impact.terrains import FAMILIES
            terrain = raw_env.scene.terrain
            cols = array(terrain.terrain_types)
            rows = array(terrain.terrain_levels)
            terrain_metadata = {
                "families": list(FAMILIES), "heldout_family_ids": [6],
                "family_id": (cols // 2).tolist(), "tile_row": rows.tolist(), "tile_col": cols.tolist(),
                "tile_id": (rows * 14 + cols).tolist(),
                "origins": array(terrain.env_origins).tolist(),
                "initial_state_jitter": "xy +/-0.25m, yaw +/-0.1rad, joint position multiplier .95..1.05, joint velocity +/-0.1rad/s",
                "tile_size_m": [12, 12], "episode_s": 6.0,
                "note": "4096 robots share 84 tiles; independent collection seeds define separate geometry splits"
            }
            np.savez(args.output / "terrain_assignment.npz", family_id=cols // 2, tile_id=rows * 14 + cols,
                     tile_row=rows, tile_col=cols, origins=array(terrain.env_origins))
        trace_ids = torch.arange(min(8, args.num_envs), device=env.device)
        torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
        start = time.perf_counter()

        def flush():
            if not pending:
                return
            data = {key: np.stack([row[key] for row in pending]) for key in pending[0]}
            name = f"shard_{len(shards):05d}.npz"
            path = args.output / name
            np.savez(path, **data)
            shards.append({"file": name, "steps": len(pending), "sha256": sha256(path),
                           "bytes": path.stat().st_size})
            pending.clear()

        with torch.inference_mode():
            for step in range(args.steps):
                pre = env.get_observations()
                reconstructed = stack_frame(pre[:, :45], reconstructed, 5, reset_ids)
                history_error = max(history_error, (reconstructed - pre).abs().max().item())
                if history_error != 0:
                    raise AssertionError("Saved raw frames cannot reconstruct exact deployed history")
                action = policy(pre) if policy is not None else torch.zeros(args.num_envs, 12, device=env.device)
                if action.shape != (args.num_envs, 12) or not torch.isfinite(action).all():
                    raise ValueError("Invalid frozen-policy action")
                row = {"policy_step": np.int64(step), "episode_id": array(episode),
                       "episode_step": array(age), "pre_frame": array(pre[:, :45]),
                       "action": array(action)}
                if terrain_metadata is not None:
                    row["root_xy_from_origin"] = array(
                        robot.data.root_pos_w.torch[:, :2] - raw_env.scene.terrain.env_origins[:, :2])
                if args.force_timeout_step == step:
                    env.episode_length_buf[:] = env.max_episode_length - 1
                if observer is not None:
                    observer.begin()
                post, next_critic, _, done, infos, reset_ids, terminal_critic = env.step(action)
                if args.him_supervision:
                    next48 = next_critic[:, :48].clone()
                    next48[reset_ids] = terminal_critic[:, :48]
                    if len(reset_ids):
                        expected_terminal = infos["final_obs"]["critic"][reset_ids, :48]
                        if not torch.equal(next48[reset_ids], expected_terminal):
                            raise AssertionError("HIM targets must use pre-reset terminal critic")
                    if not torch.isfinite(next48).all():
                        raise ValueError("Non-finite original HIM supervision")
                    row["him_next_critic48"] = array(next48)
                if observer is not None:
                    samples = observer.end()
                    sampling_stride = max(1, round(.005 / args.physics_dt))
                    row['physics_observations'] = array(samples[:, sampling_stride-1::sampling_stride])
                final = infos.get("final_obs", {}).get("policy")
                row["post_frame"] = array(completed_interval_frame(post, reset_ids, final))
                row["done"] = array(done)
                row["terminated"] = array(raw_env.reset_terminated)
                row["truncated"] = array(raw_env.reset_time_outs)
                # Derivative targets need a pre-interval history within this episode.
                row["increment_valid"] = array(age * window.decimation >= window.rise_steps)
                if not args.baseline_observer:
                    history = raw_env.impact_normal_history_w
                    targets = contact_window_targets(history, window)
                    row["normal_forces_w"] = array(history)
                    row.update({key: array(value) for key, value in targets.items()})
                    captured_terminals += int(raw_env.impact_terminal_captured.sum().item())
                    # Bounded summaries; exact quantiles remain available in shards.
                    summary.append({"mean_peak_n": float(targets["normal_peak_n"].mean()),
                                    "max_peak_n": float(targets["normal_peak_n"].max()),
                                    "mean_integral_ns": float(targets["normal_integral_ns"].mean()),
                                    "valid_increment_rows": int(row["increment_valid"].sum())})
                pending.append(row)
                trace.append({"root": array(robot.data.root_state_w.torch[trace_ids]),
                              "q": array(robot.data.joint_pos.torch[trace_ids]),
                              "dq": array(robot.data.joint_vel.torch[trace_ids]),
                              "action": row["action"][array(trace_ids)],
                              "done": row["done"][array(trace_ids)]})
                episode[reset_ids] += 1
                age += 1
                age[reset_ids] = 0
                reset_count += len(reset_ids)
                if len(pending) == args.shard_steps:
                    flush()
                if (step + 1) % 100 == 0:
                    print(f"IMPACT_COLLECTION_PROGRESS step={step + 1}/{args.steps} resets={reset_count}", flush=True)
            flush()
        elapsed_s = time.perf_counter() - start
        np.savez(args.output / "behavior_trace.npz", **{
            key: np.stack([row[key] for row in trace]) for key in trace[0]})
        if sha256(policy_path) != before_policy_hash:
            raise AssertionError("Frozen policy file changed")
        if not args.baseline_observer:
            if raw_env.impact_reference_max_error > 1e-5 or raw_env.impact_history_overlap_max_error != 0:
                raise AssertionError("Contact sensor order/history continuity check failed")
            if reset_count != captured_terminals:
                raise AssertionError("Terminal contact data were lost")
        from isaaclab.envs import ManagerBasedRLEnv
        import inspect
        lab_source = Path(inspect.getfile(ManagerBasedRLEnv))
        version_file = lab_source.parents[4] / "VERSION"
        report = {
            "status": "completed", "stage": "A0_preparation_collection_not_model_training",
            "label_validation": "Raw proxies only; downstream label definition and validation are versioned separately",
            "baseline_contract_tests_passed": baseline_contract_tests,
            "num_envs": args.num_envs, "steps": args.steps, "transitions": args.num_envs * args.steps,
            "scenario": args.scenario, "seed": args.seed, "velocity": list(args.velocity),
            "velocity_range_x": args.velocity_range_x, "episode_length_s": cfg.episode_length_s,
            "terrain": terrain_metadata,
            "drop_height_range_m": args.drop_height_range,
            "randomization": args.randomization, "observation_noise": args.observation_noise,
            "research_observations": args.research_observations,
            "him_supervision": args.him_supervision,
            "solver_iterations": [cfg.scene.robot.spawn.articulation_props.solver_position_iteration_count,
                                  cfg.scene.robot.spawn.articulation_props.solver_velocity_iteration_count],
            "solver_global_bounds": [cfg.sim.physics.min_position_iteration_count,cfg.sim.physics.max_position_iteration_count,
                                     cfg.sim.physics.min_velocity_iteration_count,cfg.sim.physics.max_velocity_iteration_count],
            "physics_observation_schema": None if observer is None else {
                "shape": "[time,env,200Hz_sample,60] float32; before reset",
                "sample_dt": max(.005, args.physics_dt),
                "channels": "clean policy45, root-COM ideal specific force3 in root frame, actual applied torque12",
                "imu_limit": "ideal simulator COM IMU; no hardware bandwidth, placement, bias or saturation model",
                "base_body": robot.body_names[0], "changes_policy_noise_rng": False},
            "baseline_observer": args.baseline_observer, "forced_timeout_step": args.force_timeout_step,
            "action_source": "frozen_HIMEstimator_and_actor_JIT" if walk else "zero_action_nominal_PD",
            "window": asdict(window), "history_length": window.history_length,
            "force_time_offsets_from_input_s": list(window.offsets_s),
            "force_order": f"oldest first; last {window.decimation} samples belong to the completed action interval",
            "foot_names": list(FOOT_NAMES), "joint_names": list(robot.joint_names),
            "policy": str(policy_path), "policy_sha256": before_policy_hash,
            "checkpoint_sha256": baseline["checkpoint_sha256"],
            "reset_count": reset_count, "terminal_traces_captured": captured_terminals,
            "history_reconstruction_max_error": history_error,
            "reference_contact_max_error": None if args.baseline_observer else raw_env.impact_reference_max_error,
            "history_overlap_max_error": None if args.baseline_observer else raw_env.impact_history_overlap_max_error,
            "startup_s": startup_s, "collection_and_io_s": elapsed_s,
            "transitions_per_second_including_io": args.num_envs * args.steps / elapsed_s,
            "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
            "memory_note": "Torch allocator excludes PhysX, Warp, Kit and driver allocations",
            "python": platform.python_version(), "torch": torch.__version__,
            "isaacsim": importlib.metadata.version("isaacsim"),
            "isaaclab_source_version": version_file.read_text().strip() if version_file.exists() else None,
            "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
            "source_sha256": {str(path.relative_to(REPO) if path.is_relative_to(REPO) else path): sha256(path)
                              for path in source_paths},
            "lab_step_source_sha256": sha256(lab_source),
            "shards": shards, "step_summary": summary,
            "behavior_trace_sha256": sha256(args.output / "behavior_trace.npz"),
            "schema": {
                "layout": "time-major [step, env, ...]; global env indices are 0..num_envs-1",
                "input_time": "policy_step * policy_dt, before the recorded action",
                "post_frame": "completed interval's frame, terminal replaced with pre-reset final_obs",
                "history": "current first + five past frames; reconstruct from pre_frame and episode_id; carry across shards",
                "detection": "post_frame concatenated with pre_history[:225], no future measurements",
                "forecast": "pre_history and optionally this action; not future actions",
                "terminal": "keep complete terminal interval; never connect horizons across done",
                "increment_valid": "false until enough pre-interval force samples exist in the episode",
                "force_semantics": collection["force_semantics"],
                "integral": "normal-resultant magnitude integral, includes support loading; not isolated impact impulse"
            }
        }
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        (args.output / "INCOMPLETE").unlink()
        print("IMPACT_COLLECTION_OK", json.dumps({key: report[key] for key in (
            "num_envs", "steps", "transitions", "reset_count", "history_reconstruction_max_error",
            "reference_contact_max_error", "history_overlap_max_error", "collection_and_io_s")}), flush=True)
    finally:
        env.close()


try:
    main()
except BaseException:
    import traceback
    traceback.print_exc()
    launcher.app.close(exit_code=1)
else:
    launcher.app.close()
