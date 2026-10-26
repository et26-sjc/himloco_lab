"""Lab-native metadata bound to an external contract and exact JIT checksum.

Hardware alignment/threshold records are inherited from the same robot's Gym
deployment. They do not establish hardware performance of the new Lab policy.
"""

import json
import math
from pathlib import Path
import re

from .contract import (HISTORY_FRAMES, POLICY_DT, POLICY_PROFILE, TRAINING_JOINT_NAMES,
                       JointMap, validate_rotation_matrix, validate_deployment_contract)
from .policy import sha256_file

_TOP_KEYS = {"schema_version", "source_commit", "policy_profile", "deployment_contract",
             "model", "alignment", "runtime"}
_MODEL_KEYS = {"path", "sha256", "training_dof_names", "training_config_reference"}
_ALIGNMENT_KEYS = {"attitude_source", "sensor_to_body_rotation", "verified",
                   "verification_reference", "joint_mapping_verified", "joint_mapping_reference"}
_RUNTIME_KEYS = {
    "policy_dt", "state_timeout_s", "command_timeout_s", "rc_timeout_s",
    "calibration_duration_s", "history_warmup_frames", "action_ramp_s",
    "startup_joint_margin_rad", "calibration_gain_ramp_s", "calibration_target_inset_rad",
    "max_abs_command", "command_axis_signs", "command_deadzone", "max_tilt_rad",
    "max_joint_target_step_rad", "max_tracking_error_rad", "calibration_tolerance_rad",
    "calibration_settle_timeout_s", "max_cycle_lateness_s", "startup_timeout_s",
    "ready_timeout_s", "safety_reviewed", "safety_review_reference", "lcm_url",
}
_RUNTIME_OPTIONAL_KEYS = {"tracking_error_timeout_s", "max_instant_tracking_error_rad"}


def _keys(value, expected, label):
    if not isinstance(value, dict):
        raise ValueError("{} must be a JSON object".format(label))
    actual = set(value)
    if label == "configuration":
        actual.discard("_config_path")
    if actual != expected:
        raise ValueError("{} keys mismatch: missing {}; unknown {}".format(
            label, sorted(expected - actual), sorted(actual - expected)))


def _text_or_null(value, label):
    if value is not None and (not isinstance(value, str) or not value.strip()):
        raise ValueError("{} must be a nonempty string or null".format(label))


def _number(value, label, positive=True):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("{} must be a finite number".format(label))
    if positive and value <= 0:
        raise ValueError("{} must be positive".format(label))
    return value


def readiness_errors(config):
    """Return missing recorded evidence; this does not itself verify hardware."""
    errors = []
    model = config["model"]
    alignment = config["alignment"]
    runtime = config["runtime"]
    if model["sha256"] is None:
        errors.append("model.sha256 is unset")
    if model["training_dof_names"] is None:
        errors.append("model.training_dof_names must come from the actual training run")
    if model["training_config_reference"] is None:
        errors.append("model.training_config_reference is unset")
    if not alignment["verified"] or alignment["verification_reference"] is None:
        errors.append("IMU attitude/gyro frame alignment requires a verification record")
    if not alignment["joint_mapping_verified"] or alignment["joint_mapping_reference"] is None:
        errors.append("physical joint identities and signs require a verification record")
    if not runtime["safety_reviewed"] or runtime["safety_review_reference"] is None:
        errors.append("runtime thresholds require an explicit review record for this robot")
    return errors


def validate_config(config, require_ready=False):
    _keys(config, _TOP_KEYS, "configuration")
    if type(config["schema_version"]) is not int or config["schema_version"] != 2:
        raise ValueError("schema_version must be 2 for the Lab-native policy interface")
    if config["policy_profile"] != POLICY_PROFILE:
        raise ValueError("policy_profile must be " + POLICY_PROFILE)
    if not isinstance(config["source_commit"], str) or not re.fullmatch(r"[0-9a-f]{40}", config["source_commit"]):
        raise ValueError("source_commit must record the actual Lab training Git SHA")
    descriptor = config["deployment_contract"]
    _keys(descriptor, {"path", "sha256"}, "deployment_contract")
    if not isinstance(descriptor["path"], str) or not descriptor["path"].strip():
        raise ValueError("deployment_contract.path must be a nonempty string")
    if not isinstance(descriptor["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", descriptor["sha256"]):
        raise ValueError("deployment_contract.sha256 must be a recorded SHA-256")
    model = config["model"]
    alignment = config["alignment"]
    runtime = config["runtime"]
    _keys(model, _MODEL_KEYS, "model")
    _keys(alignment, _ALIGNMENT_KEYS, "alignment")
    runtime_keys = (_RUNTIME_KEYS | (set(runtime) & _RUNTIME_OPTIONAL_KEYS)
                    if isinstance(runtime, dict) else _RUNTIME_KEYS)
    _keys(runtime, runtime_keys, "runtime")
    if not isinstance(model["path"], str) or not model["path"].strip():
        raise ValueError("model.path must be a nonempty string")
    if model["sha256"] is not None and (not isinstance(model["sha256"], str) or
            re.fullmatch(r"[0-9a-fA-F]{64}", model["sha256"]) is None):
        raise ValueError("model.sha256 must be null or a 64-character SHA-256")
    if model["training_dof_names"] is not None:
        JointMap(model["training_dof_names"])
        if tuple(model["training_dof_names"]) != TRAINING_JOINT_NAMES:
            raise ValueError("Lab policy requires the recorded Lab training joint order")
    _text_or_null(model["training_config_reference"], "model.training_config_reference")
    if alignment["attitude_source"] not in ("rpy", "quaternion_wxyz"):
        raise ValueError("alignment.attitude_source must be rpy or quaternion_wxyz")
    validate_rotation_matrix(alignment["sensor_to_body_rotation"])
    for key in ("verified", "joint_mapping_verified"):
        if type(alignment[key]) is not bool:
            raise ValueError("alignment.{} must be boolean".format(key))
    for key in ("verification_reference", "joint_mapping_reference"):
        _text_or_null(alignment[key], "alignment." + key)
    if type(runtime["safety_reviewed"]) is not bool:
        raise ValueError("runtime.safety_reviewed must be boolean")
    _text_or_null(runtime["safety_review_reference"], "runtime.safety_review_reference")
    if not isinstance(runtime["lcm_url"], str) or not runtime["lcm_url"].startswith("udpm://"):
        raise ValueError("runtime.lcm_url must be an explicit udpm:// LCM URL")
    for key in ("policy_dt", "state_timeout_s", "command_timeout_s", "rc_timeout_s",
                "calibration_duration_s", "action_ramp_s", "max_tilt_rad",
                "startup_joint_margin_rad", "calibration_gain_ramp_s", "calibration_target_inset_rad",
                "max_tracking_error_rad", "calibration_tolerance_rad",
                "calibration_settle_timeout_s", "max_cycle_lateness_s", "startup_timeout_s",
                "ready_timeout_s"):
        _number(runtime[key], "runtime." + key)
    # Older configs retain their immediate tracking stop. The optional delay
    # applies only in RAMP/RUNNING, separately for each physical joint.
    tracking_timeout = _number(runtime.get("tracking_error_timeout_s", 0.0),
                               "runtime.tracking_error_timeout_s", positive=False)
    if tracking_timeout < 0:
        raise ValueError("tracking_error_timeout_s must be nonnegative")
    instant_limit = _number(
        runtime.get("max_instant_tracking_error_rad", runtime["max_tracking_error_rad"]),
        "runtime.max_instant_tracking_error_rad")
    if instant_limit < runtime["max_tracking_error_rad"]:
        raise ValueError("max_instant_tracking_error_rad must be >= max_tracking_error_rad")
    # A null value explicitly disables only the added target-delta stop.
    # Mechanical bounds, tracking, tilt, freshness and stop buttons still apply.
    if runtime["max_joint_target_step_rad"] is not None:
        _number(runtime["max_joint_target_step_rad"], "runtime.max_joint_target_step_rad")
    if runtime["startup_joint_margin_rad"] > 0.2:
        raise ValueError("startup_joint_margin_rad must be <=0.2; running limits are unchanged")
    if runtime["calibration_gain_ramp_s"] > runtime["calibration_duration_s"]:
        raise ValueError("calibration_gain_ramp_s must not exceed calibration_duration_s")
    if runtime["calibration_target_inset_rad"] > 0.01:
        raise ValueError("calibration_target_inset_rad must be <=0.01")
    if not math.isclose(runtime["policy_dt"], POLICY_DT, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("runtime.policy_dt must be the source policy period 0.02 s")
    if not math.isclose(runtime["command_timeout_s"], 0.1, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("runtime.command_timeout_s must match the bridge's fixed 0.1 s watchdog")
    if runtime["state_timeout_s"] > 0.05 or runtime["rc_timeout_s"] > 0.1:
        raise ValueError("state_timeout_s must be <=0.05 and rc_timeout_s <=0.1")
    if runtime["max_cycle_lateness_s"] >= runtime["command_timeout_s"]:
        raise ValueError("max_cycle_lateness_s must be below the command watchdog timeout")
    if runtime["max_tilt_rad"] >= math.pi / 2:
        raise ValueError("max_tilt_rad must be less than pi/2")
    frames = runtime["history_warmup_frames"]
    if type(frames) is not int or frames < HISTORY_FRAMES:
        raise ValueError("history_warmup_frames must be an integer >=6")
    deadzone = _number(runtime["command_deadzone"], "runtime.command_deadzone", positive=False)
    if not 0 <= deadzone < 1:
        raise ValueError("command_deadzone must be in [0,1)")
    limits = runtime["max_abs_command"]
    if not isinstance(limits, list) or len(limits) != 3:
        raise ValueError("max_abs_command must contain three physical-unit limits")
    for value in limits:
        _number(value, "runtime.max_abs_command entry")
    signs = runtime["command_axis_signs"]
    if (not isinstance(signs, list) or len(signs) != 3 or
            any(isinstance(x, bool) or x not in (-1, 1) for x in signs)):
        raise ValueError("command_axis_signs must contain three values, each -1 or +1")
    if require_ready:
        errors = readiness_errors(config)
        if errors:
            raise ValueError("Live deployment is not ready: " + "; ".join(errors))
    return config


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key: " + key)
        result[key] = value
    return result


def load_config(path, require_ready=False):
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8-sig") as stream:
        config = json.load(stream, object_pairs_hook=_unique_object)
    validate_config(config, require_ready=require_ready)
    model_path = Path(config["model"]["path"])
    if not model_path.is_absolute():
        model_path = config_path.parent / model_path
    config["model"]["path"] = str(model_path.resolve())
    contract_path = Path(config["deployment_contract"]["path"])
    if not contract_path.is_absolute():
        contract_path = config_path.parent / contract_path
    contract_path = contract_path.resolve()
    if sha256_file(contract_path) != config["deployment_contract"]["sha256"]:
        raise ValueError("deployment contract SHA-256 mismatch")
    with contract_path.open(encoding="utf-8") as stream:
        contract = json.load(stream, object_pairs_hook=_unique_object)
    validate_deployment_contract(contract)
    if contract["provenance"]["training_git_sha"] != config["source_commit"]:
        raise ValueError("training Git SHA differs between config and deployment contract")
    if contract["provenance"]["policy_sha256"] != config["model"]["sha256"]:
        raise ValueError("policy SHA differs between config and deployment contract")
    config["deployment_contract"]["path"] = str(contract_path)
    config["_config_path"] = str(config_path)
    return config
