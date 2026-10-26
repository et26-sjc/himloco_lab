import copy
import math
import unittest

import numpy as np

from go1_lab_deploy.himloco.alignment import analyze, rotation_vector
from go1_lab_deploy.himloco.contract import SDK_JOINT_NAMES, rotation_rpy


def quaternion(r):
    # Synthetic fixture matrices are away from 180 degrees.
    w = .5 * math.sqrt(1 + np.trace(r))
    return np.array([w, (r[2, 1] - r[1, 2]) / (4*w),
                     (r[0, 2] - r[2, 0]) / (4*w),
                     (r[1, 0] - r[0, 1]) / (4*w)])


def rpy(r):
    return np.array([math.atan2(r[2, 1], r[2, 2]),
                     math.asin(np.clip(-r[2, 0], -1, 1)),
                     math.atan2(r[1, 0], r[0, 0])])


def captures(mount):
    data = {}
    labels = [("neutral", np.zeros(3))] + list({
        "roll_left_up": np.array([1., 0, 0]),
        "pitch_nose_up": np.array([0., -1, 0]),
        "yaw_left": np.array([0., 0, 1])}.items())
    for label, axis in labels:
        rows = []
        for i in range(201):
            t = i * .02
            fraction = np.clip((t - 1) / 2, 0, 1)
            rb = rotation_rpy(axis * math.radians(15) * fraction)
            rs = rb.dot(mount)
            rate = math.radians(15) / 2 if 1 < t < 3 else 0
            rows.append(dict(capture_monotonic_s=t, quaternion_wxyz=quaternion(rs),
                             rpy=rpy(rs), gyro_sensor=mount.T.dot(axis * rate),
                             q_sdk=np.zeros(12)))
        data[label] = rows
    for j, name in enumerate(SDK_JOINT_NAMES):
        rows = copy.deepcopy(data["neutral"])
        for row in rows:
            row["q_sdk"][j] = .15 * np.clip((row["capture_monotonic_s"] - 1) / 2, 0, 1)
        data["joint_" + name] = rows
    return data


class AlignmentTests(unittest.TestCase):
    def setUp(self):
        self.mount = rotation_rpy([.3, -.2, .7])
        self.data = captures(self.mount)

    def test_mount_axes_and_all_joints(self):
        r = analyze(self.data, level_confirmed=True)
        self.assertTrue(r["imu_verified"])
        self.assertTrue(r["joint_mapping_verified"])
        np.testing.assert_allclose(r["candidate_sensor_to_body_rotation"], self.mount, atol=1e-6)
        self.assertLess(r["neutral_tilt_deg"], 1e-5)

    def test_physical_level_is_required(self):
        self.assertFalse(analyze(self.data)["imu_verified"])

    def test_quaternion_double_cover(self):
        for rows in self.data.values():
            for row in rows[::2]:
                row["quaternion_wxyz"] *= -1
        self.assertTrue(analyze(self.data, True)["imu_verified"])

    def test_rpy_mismatch_rejected(self):
        self.data["neutral"][40]["rpy"][0] += .1
        self.assertFalse(analyze(self.data, True)["imu_verified"])

    def test_missing_rotation_rejected(self):
        del self.data["yaw_left"]
        self.assertFalse(analyze(self.data, True)["imu_verified"])

    def test_no_excitation_rejected(self):
        self.data["pitch_nose_up"] = copy.deepcopy(self.data["neutral"])
        self.assertFalse(analyze(self.data, True)["imu_verified"])

    def test_gyro_sign_rejected(self):
        for row in self.data["pitch_nose_up"]:
            row["gyro_sensor"] *= -1
        self.assertFalse(analyze(self.data, True)["imu_verified"])

    def test_joint_index_and_sign_rejected(self):
        label = "joint_" + SDK_JOINT_NAMES[0]
        for row in self.data[label]:
            row["q_sdk"][0] *= -1
        self.assertFalse(analyze(self.data, True)["joint_mapping_verified"])

    def test_wrong_joint_is_rejected(self):
        label = "joint_" + SDK_JOINT_NAMES[0]
        for row in self.data[label]:
            row["q_sdk"][[0, 1]] = row["q_sdk"][[1, 0]]
        self.assertFalse(analyze(self.data, True)["joint_mapping_verified"])

    def test_moving_neutral_is_rejected_even_with_zero_mean_gyro(self):
        for index, row in enumerate(self.data["neutral"]):
            row["gyro_sensor"][0] = .1 if index % 2 else -.1
        self.assertFalse(analyze(self.data, True)["imu_verified"])

    def test_bad_time_or_nonfinite_is_rejected(self):
        self.data["neutral"][10]["capture_monotonic_s"] = .5
        with self.assertRaisesRegex(ValueError, "time order"):
            analyze(self.data, True)

    def test_near_pi_motion_rejected(self):
        with self.assertRaisesRegex(ValueError, "180"):
            rotation_vector(rotation_rpy([math.pi, 0, 0]))


if __name__ == "__main__":
    unittest.main()
