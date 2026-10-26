"""Offline analysis of operator-labelled, passive alignment captures.

Labels describe physical motions, not expected numerical encoder/IMU outputs.
No communication objects, policy loading, or configuration writes occur here.
"""
import math
import numpy as np

from .contract import (SDK_JOINT_NAMES, rotation_rpy,
                       rotation_quaternion_wxyz, validate_rotation_matrix)


def rotation_vector(rotation):
    r = validate_rotation_matrix(rotation)
    angle = math.acos(float(np.clip((np.trace(r) - 1.0) / 2.0, -1, 1)))
    skew = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])
    if angle < 1e-7:
        return .5 * skew
    if angle > math.pi - .01:
        raise ValueError("Near-180-degree motion is unsuitable for alignment")
    return angle / (2.0 * math.sin(angle)) * skew


def mean_rotation(rows):
    quats = np.asarray([r["quaternion_wxyz"] for r in rows], dtype=float)
    # Account for q and -q being the same attitude before averaging.
    quats *= np.where(quats.dot(quats[0]) < 0, -1.0, 1.0)[:, None]
    mean = np.mean(quats, axis=0)
    norm = np.linalg.norm(mean)
    if not np.isfinite(norm) or norm < .9:
        raise ValueError("Attitude capture has excessive dispersion")
    return rotation_quaternion_wxyz(mean / norm)


def endpoints(rows):
    if len(rows) < 40:
        raise ValueError("At least 40 fresh samples are required")
    span = rows[-1]["capture_monotonic_s"] - rows[0]["capture_monotonic_s"]
    if span < 2:
        raise ValueError("Capture must span at least 2 seconds")
    start, end = rows[0]["capture_monotonic_s"], rows[-1]["capture_monotonic_s"]
    return ([r for r in rows if r["capture_monotonic_s"] <= start + .7],
            [r for r in rows if r["capture_monotonic_s"] >= end - .7])


def validate_capture(rows):
    endpoints(rows)
    times = np.asarray([r["capture_monotonic_s"] for r in rows], dtype=float)
    if not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0) or np.any(np.diff(times) >= .1):
        raise ValueError("Capture has invalid time order or a gap of at least 100 ms")
    for row in rows:
        for key, size in (("q_sdk", 12), ("gyro_sensor", 3), ("rpy", 3), ("quaternion_wxyz", 4)):
            v = np.asarray(row[key], dtype=float)
            if v.shape != (size,) or not np.all(np.isfinite(v)):
                raise ValueError("Invalid capture field: " + key)
        rotation_quaternion_wxyz(row["quaternion_wxyz"])


def static_attitude_metrics(rows):
    mean = mean_rotation(rows)
    dispersion = max(np.linalg.norm(rotation_vector(
        mean.T.dot(rotation_quaternion_wxyz(r["quaternion_wxyz"])))) for r in rows)
    gyro_rms = float(np.sqrt(np.mean(np.sum(np.asarray(
        [r["gyro_sensor"] for r in rows]) ** 2, axis=1))))
    return dict(passed=bool(dispersion < math.radians(1) and gyro_rms < .03),
                attitude_dispersion_deg_max=float(np.degrees(dispersion)),
                gyro_rms_rad_s=gyro_rms)


def static_attitude(rows):
    return static_attitude_metrics(rows)["passed"]


def analyze(captures, level_confirmed=False):
    """Require three independent labelled rotations and all 12 positive joints.

    A fitted matrix is a candidate mounting transform. Its dynamic gyro checks
    and the independent, physically level neutral capture must also pass.
    """
    result = dict(schema_version=1, imu_verified=False, joint_mapping_verified=False,
                  level_confirmed=bool(level_confirmed), missing=[], failures=[],
                  pose_checks={}, joint_checks={})
    for rows in captures.values():
        validate_capture(rows)
    all_rows = [r for rows in captures.values() for r in rows]
    disagreements = []
    for row in all_rows:
        rq = rotation_quaternion_wxyz(row["quaternion_wxyz"])
        rr = rotation_rpy(row["rpy"])
        disagreements.append(np.linalg.norm(rotation_vector(rq.T.dot(rr))))
    if disagreements:
        result["rpy_quaternion_median_error_deg"] = float(np.degrees(np.median(disagreements)))
        result["rpy_quaternion_max_error_deg"] = float(np.degrees(max(disagreements)))
    attitude_ok = bool(disagreements and max(disagreements) < math.radians(3))
    if not attitude_ok:
        result["failures"].append("RPY/quaternion disagreement or no attitude samples")

    # Positive right-handed body rotations: roll left side up, pitch nose DOWN,
    # yaw nose LEFT. The user labels nose UP, hence the negative pitch axis.
    axes = {"roll_left_up": np.array([1., 0, 0]),
            "pitch_nose_up": np.array([0., -1, 0]),
            "yaw_left": np.array([0., 0, 1])}
    measured, desired, motions = [], [], {}
    for label, expected in axes.items():
        if label not in captures:
            result["missing"].append(label)
            continue
        first, last = endpoints(captures[label])
        v = rotation_vector(mean_rotation(first).T.dot(mean_rotation(last)))
        angle = float(np.linalg.norm(v))
        if not math.radians(5) <= angle <= math.radians(40):
            result["failures"].append(label + ": net rotation must be 5 to 40 degrees")
            continue
        measured.append(v / angle)
        desired.append(expected)
        motions[label] = (v, first, last)

    if len(measured) == 3:
        source = np.stack(measured, axis=1)
        target = np.stack(desired, axis=1)
        u, _, vt = np.linalg.svd(target.dot(source.T))
        rotation = u.dot(np.diag([1., 1., np.linalg.det(u.dot(vt))])).dot(vt)
        result["candidate_sensor_to_body_rotation"] = rotation.tolist()
        pose_ok = True
        for label, (v, first, last) in motions.items():
            rows = captures[label]
            transformed_axis = rotation.dot(v / np.linalg.norm(v))
            axis_error = math.acos(float(np.clip(transformed_axis.dot(axes[label]), -1, 1)))
            gyro_bias = np.mean([r["gyro_sensor"] for r in first], axis=0)
            integrated = np.zeros(3)
            for before, after in zip(rows[:-1], rows[1:]):
                dt = after["capture_monotonic_s"] - before["capture_monotonic_s"]
                if not 0 < dt < .1:
                    pose_ok = False
                    continue
                integrated += (.5 * (np.asarray(before["gyro_sensor"]) +
                                      np.asarray(after["gyro_sensor"])) - gyro_bias) * dt
            gyro_body = rotation.dot(integrated)
            magnitude = float(np.linalg.norm(gyro_body))
            cosine = float(gyro_body.dot(axes[label]) / max(magnitude, 1e-12))
            ratio = magnitude / np.linalg.norm(v)
            start_static = static_attitude_metrics(first)
            end_static = static_attitude_metrics(last)
            endpoints_static = start_static["passed"] and end_static["passed"]
            good = (axis_error < math.radians(10) and cosine > .9 and .5 < ratio < 1.5
                    and endpoints_static)
            pose_ok = pose_ok and good
            result["pose_checks"][label] = dict(
                passed=bool(good), endpoints_static=bool(endpoints_static),
                start_static_check=start_static, end_static_check=end_static,
                raw_sensor_rotation_axis=(v / np.linalg.norm(v)).tolist(),
                identity_mount_axis_error_deg=float(np.degrees(math.acos(float(np.clip(
                    (v / np.linalg.norm(v)).dot(axes[label]), -1, 1))))),
                angle_deg=float(np.degrees(np.linalg.norm(v))),
                axis_error_deg=float(np.degrees(axis_error)),
                integrated_gyro_body_rad=gyro_body.tolist(),
                gyro_axis_cosine=cosine, gyro_to_attitude_angle_ratio=float(ratio))
        if "neutral" in captures:
            rows = captures["neutral"]
            sensor_gravity = mean_rotation(rows).T.dot([0., 0, -1.])
            gravity = rotation.dot(sensor_gravity)
            tilt = math.acos(float(np.clip(-gravity[2], -1, 1)))
            result["neutral_gravity_body"] = gravity.tolist()
            result["neutral_tilt_deg"] = float(np.degrees(tilt))
            result["identity_mount_neutral_tilt_deg"] = float(np.degrees(math.acos(
                float(np.clip(-sensor_gravity[2], -1, 1)))))
            result["neutral_gyro_norm_rad_s"] = float(np.linalg.norm(
                np.mean([r["gyro_sensor"] for r in rows], axis=0)))
            static = static_attitude(rows)
            result["neutral_static"] = bool(static)
            result["imu_verified"] = bool(attitude_ok and pose_ok and level_confirmed
                                            and tilt < math.radians(2) and static)
        else:
            result["missing"].append("neutral")
    if not level_confirmed:
        result["missing"].append("physical level reference confirmation")

    for index, name in enumerate(SDK_JOINT_NAMES):
        label = "joint_" + name
        if label not in captures:
            result["missing"].append(label)
            continue
        first, last = endpoints(captures[label])
        delta = (np.mean([r["q_sdk"] for r in last], axis=0) -
                 np.mean([r["q_sdk"] for r in first], axis=0))
        others = np.abs(np.delete(delta, index))
        largest = int(np.argmax(np.abs(delta)))
        passed = (largest == index and .06 < delta[index] < .6 and
                  float(max(others)) < max(.04, .35 * abs(delta[index])))
        result["joint_checks"][name] = dict(passed=bool(passed), delta_q_sdk_rad=delta.tolist(),
                                            largest_changed_joint=SDK_JOINT_NAMES[largest],
                                            expected_positive_delta_rad=float(delta[index]))
    checks = result["joint_checks"]
    result["joint_mapping_verified"] = bool(len(checks) == 12 and all(c["passed"] for c in checks.values()))
    return result
