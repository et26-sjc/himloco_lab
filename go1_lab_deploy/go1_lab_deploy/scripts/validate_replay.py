"""CPU-only replay of synchronized native Lab samples through deployment code.

Sensor states are recorded, not simulated. Each environment has its own history
and previous-action cache, fed by this tool's JIT outputs. No transport or live
runner is imported and no motor commands are published.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from go1_lab_deploy.himloco.config import load_config
from go1_lab_deploy.himloco.contract import (
    JointMap, ObservationHistory, SDK_JOINT_NAMES, action_to_target,
    build_observation, default_positions, joint_limits, projected_gravity,
    validate_rotation_matrix,
)
from go1_lab_deploy.himloco.policy import JitPolicy, sha256_file


class ErrorStats:
    def __init__(self):
        self.maximum = 0.0
        self.total = 0.0
        self.count = 0

    def add(self, actual, expected):
        actual = np.asarray(actual, dtype=np.float64)
        expected = np.asarray(expected, dtype=np.float64)
        if actual.shape != expected.shape or not np.isfinite(actual).all() or not np.isfinite(expected).all():
            raise ValueError("Comparison requires identically shaped finite arrays")
        error = np.abs(actual - expected)
        self.maximum = max(self.maximum, float(np.max(error)))
        self.total += float(np.sum(error))
        self.count += error.size

    def to_dict(self):
        return dict(max_abs_error=self.maximum, mean_abs_error=self.total / self.count,
                    compared_values=self.count)


def validate(config_path, reference_path):
    config_path, reference_path = Path(config_path).resolve(), Path(reference_path).resolve()
    metadata_path = reference_path.with_name("metadata.json")
    config = load_config(config_path)
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("reference_sha256") != sha256_file(reference_path):
        raise ValueError("Reference data SHA-256 does not match metadata")
    if metadata.get("policy_sha256") != config["model"]["sha256"]:
        raise ValueError("Reference was recorded with another policy")
    contract_path = Path(config["deployment_contract"]["path"])
    contract = json.loads(contract_path.read_text())
    if metadata.get("checkpoint_sha256") != contract["provenance"]["checkpoint_sha256"]:
        raise ValueError("Reference and deployment checkpoint provenance differ")
    if config["alignment"]["attitude_source"] != "quaternion_wxyz":
        raise ValueError("This synchronized reference records quaternion_wxyz attitude")
    mapping = JointMap(config["model"]["training_dof_names"])
    nominal = default_positions(mapping.policy_names)
    rotation = validate_rotation_matrix(config["alignment"]["sensor_to_body_rotation"])
    lower, upper = joint_limits(mapping.policy_names)
    policy = JitPolicy(config["model"]["path"], config["model"]["sha256"])
    fields = {
        "step": (), "env_id": (), "episode_id": (), "history_reset": (),
        "q_sdk": (12,), "dq_sdk": (12,), "quaternion_wxyz": (4,), "gyro_sensor": (3,),
        "commands": (3,), "previous_action_lab": (12,), "expected_frame": (45,),
        "expected_history": (270,), "raw_action_lab": (12,), "expected_target_lab": (12,),
        "done_after_action": (),
    }
    slices = {"commands": slice(0, 3), "base_ang_vel": slice(3, 6),
              "projected_gravity": slice(6, 9), "joint_pos_rel": slice(9, 21),
              "joint_vel_rel": slice(21, 33), "last_action": slice(33, 45)}
    errors = {name: ErrorStats() for name in ("frame", "history", "raw_action_lab", "target_lab",
              "target_sdk", "previous_action_cache", "sdk_q_roundtrip", "sdk_dq_roundtrip")}
    errors.update({"frame/" + name: ErrorStats() for name in slices})
    histories, previous, last_step, episode, last_done = {}, {}, {}, {}, {}
    latencies, targets, positions = [], [], []
    lower_violations = np.zeros(12, dtype=np.int64)
    upper_violations = np.zeros(12, dtype=np.int64)
    lower_touches = np.zeros(12, dtype=np.int64)
    upper_touches = np.zeros(12, dtype=np.int64)
    reset_count, bounds_violation_samples = 0, 0
    with np.load(reference_path, allow_pickle=False) as data:
        count = len(data["env_id"])
        if count < 1 or count != metadata["samples"]:
            raise ValueError("Empty reference or metadata sample-count mismatch")
        for name, shape in fields.items():
            if data[name].shape != (count,) + shape or not np.isfinite(data[name]).all():
                raise ValueError("Invalid synchronized reference field: " + name)
        if tuple(data["lab_joint_names"]) != mapping.policy_names:
            raise ValueError("Reference Lab joint names do not match the deployment policy")
        if tuple(data["sdk_joint_names"]) != SDK_JOINT_NAMES:
            raise ValueError("Reference SDK joint names do not match the deployment transport")
        np.testing.assert_array_equal(data["default_joint_pos_lab"], nominal)
        np.testing.assert_array_equal(data["default_joint_vel_lab"], np.zeros(12))
        for index in range(count):
            env = int(data["env_id"][index])
            step = int(data["step"][index])
            current_episode = int(data["episode_id"][index])
            reset = bool(data["history_reset"][index])
            if env not in histories:
                if not reset:
                    raise ValueError("First sample for an environment must reset history")
                histories[env] = ObservationHistory()
            else:
                if step != last_step[env] + 1 or reset != last_done[env]:
                    raise ValueError("Samples must be chronological and preserve episode boundaries")
                if current_episode != episode[env] + int(reset):
                    raise ValueError("Episode counter does not match the reset boundary")
            if reset:
                histories[env].reset()
                previous[env] = np.zeros(12, dtype=np.float32)
                reset_count += 1
            errors["previous_action_cache"].add(previous[env], data["previous_action_lab"][index])
            q = mapping.to_policy(data["q_sdk"][index])
            dq = mapping.to_policy(data["dq_sdk"][index])
            errors["sdk_q_roundtrip"].add(mapping.to_sdk(q), data["q_sdk"][index])
            errors["sdk_dq_roundtrip"].add(mapping.to_sdk(dq), data["dq_sdk"][index])
            gravity = projected_gravity(data["quaternion_wxyz"][index], "quaternion_wxyz", rotation)
            omega = rotation.dot(data["gyro_sensor"][index])
            frame = build_observation(data["commands"][index], omega, gravity, q, dq, previous[env], nominal)
            history = histories[env].push(frame)
            errors["frame"].add(frame, data["expected_frame"][index])
            errors["history"].add(history[0], data["expected_history"][index])
            for name, section in slices.items():
                errors["frame/" + name].add(frame[section], data["expected_frame"][index, section])
            started = time.perf_counter()
            action = policy(history)
            latencies.append((time.perf_counter() - started) * 1000.)
            target = action_to_target(action, nominal)
            target_sdk = mapping.to_sdk(target)
            errors["raw_action_lab"].add(action, data["raw_action_lab"][index])
            errors["target_lab"].add(target, data["expected_target_lab"][index])
            expected_sdk = data["expected_target_lab"][index][mapping.sdk_to_policy]
            errors["target_sdk"].add(target_sdk, expected_sdk)
            below, above = target < lower, target > upper
            lower_violations += below
            upper_violations += above
            lower_touches += np.isclose(target, lower, rtol=0, atol=1e-6)
            upper_touches += np.isclose(target, upper, rtol=0, atol=1e-6)
            bounds_violation_samples += int(np.any(below | above))
            targets.append(target)
            positions.append(q)
            previous[env] = action.copy()
            last_step[env], episode[env] = step, current_episode
            last_done[env] = bool(data["done_after_action"][index])
    if any(name == "lcm" or name.startswith("isaaclab") or name.startswith("isaacsim")
           for name in sys.modules):
        raise AssertionError("Replay must not import robot transport or Isaac runtime")
    metrics = {name: value.to_dict() for name, value in errors.items()}
    # Closed history uses this tool's previous inferred actions, so its numerical
    # roundoff propagates through later observations. Keep that distinct from
    # physical-state fields which have no recurrent policy error.
    tolerances = {name: (2e-5 if name in ("raw_action_lab", "target_lab", "target_sdk",
                                        "previous_action_cache", "frame/last_action", "frame", "history")
                        else 2e-6) for name in errors}
    failed = [name for name in metrics if metrics[name]["max_abs_error"] > tolerances[name]]
    targets, positions = np.asarray(targets), np.asarray(positions)
    ranges = []
    for joint, name in enumerate(mapping.policy_names):
        ranges.append(dict(name=name, minimum_target_rad=float(targets[:, joint].min()),
            maximum_target_rad=float(targets[:, joint].max()),
            minimum_measured_rad=float(positions[:, joint].min()), maximum_measured_rad=float(positions[:, joint].max()),
            deployment_lower_rad=float(lower[joint]), deployment_upper_rad=float(upper[joint]),
            lower_violations=int(lower_violations[joint]), upper_violations=int(upper_violations[joint]),
            lower_touches=int(lower_touches[joint]), upper_touches=int(upper_touches[joint])))
    return dict(status="passed" if not failed else "failed", failed_comparisons=failed,
        timestamp_utc=datetime.now(timezone.utc).isoformat(), samples=count, environments=len(histories),
        history_reset_samples=reset_count, finite=True, errors=metrics, absolute_tolerances=tolerances,
        cpu_inference_latency_ms=dict(p50=float(np.percentile(latencies, 50)), p95=float(np.percentile(latencies, 95)),
            maximum=float(np.max(latencies)), mean=float(np.mean(latencies)), warmup_calls=5, timed_calls=count),
        mechanical_bounds=dict(violation_samples=bounds_violation_samples,
            targets_clipped_to_deployment_limits=False, per_joint=ranges,
            interpretation="Inherited deployment bounds are diagnosed without clipping or suppressing policy output; "
                           "a live runner would reject violations. These are not native Lab target limits."),
        provenance=dict(config_path=str(config_path), config_sha256=sha256_file(config_path),
            deployment_contract_path=str(contract_path), deployment_contract_sha256=sha256_file(contract_path),
            policy_path=policy.model_path, policy_sha256=policy.sha256,
            checkpoint_sha256=contract["provenance"]["checkpoint_sha256"],
            reference_path=str(reference_path), reference_sha256=sha256_file(reference_path),
            metadata_sha256=sha256_file(metadata_path), reference_metadata=metadata,
            validator_source_sha256=sha256_file(__file__), python=platform.python_version(),
            torch=policy._torch.__version__, numpy=np.__version__, platform=platform.platform()),
        replay_semantics="Fixed recorded sensor states; independent per-environment closed observation history and "
                         "previous raw action cache. This is not a new physical trajectory or hardware validation.",
        motor_commands_published=False, isaac_runtime_imported=False, lcm_imported=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Lab deployment root configuration JSON")
    parser.add_argument("--reference", required=True, help="Synchronized native Lab reference.npz")
    parser.add_argument("--output", required=True, help="New report JSON; existing files are never overwritten")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("Select a new report path; existing replay reports are preserved")
    report = validate(args.config, args.reference)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print("LAB_DEPLOY_REPLAY_{} {}".format(report["status"].upper(), output))
    print(json.dumps(dict(errors=report["errors"], latency_ms=report["cpu_inference_latency_ms"],
                          mechanical_bound_violation_samples=report["mechanical_bounds"]["violation_samples"])))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
