import math
import unittest

import numpy as np

from go1_lab_deploy.himloco.contract import (
    ACTION_SCALE, TARGET_CLIP, COMMAND_SCALE, DEFAULT_ANGLES, KD, KP, OBS_CLIP, POLICY_DT,
    SDK_JOINT_NAMES, TRAINING_JOINT_NAMES, TRAINING_COMMIT, POLICY_PROFILE, JointMap, ObservationHistory,
    action_to_target, build_observation, default_positions,
    joint_limits, projected_gravity, rotation_quaternion_wxyz, rotation_rpy,
    validate_rotation_matrix,
)


class JointMappingTests(unittest.TestCase):
    def test_pinned_lab_policy_profile_and_named_joint_order(self):
        expected = tuple(f"{leg}_{joint}_joint" for joint in ("hip", "thigh", "calf")
                         for leg in ("FL", "FR", "RL", "RR"))
        self.assertEqual(tuple(TRAINING_JOINT_NAMES), expected)
        self.assertEqual(POLICY_PROFILE, "lab_go1_pd_v1")
        self.assertTrue(TRAINING_COMMIT.startswith("73fc02a"))
        self.assertEqual(len(TRAINING_COMMIT), 40)
        np.testing.assert_array_equal(COMMAND_SCALE, [1, 1, 1])
        self.assertEqual(KP, 40.)
        self.assertEqual(KD, 1.)
        self.assertEqual(POLICY_DT, .02)
        mapping = JointMap(TRAINING_JOINT_NAMES)
        sdk = np.arange(12, dtype=np.float32) + .125
        expected_policy = [sdk[SDK_JOINT_NAMES.index(name)] for name in expected]
        np.testing.assert_array_equal(mapping.to_policy(sdk), expected_policy)
        np.testing.assert_array_equal(mapping.to_sdk(expected_policy), sdk)

    def test_non_self_inverse_mapping_roundtrip_and_name_alignment(self):
        # A three-position cyclic shift exposes the old self-inverse-swap bug.
        names = SDK_JOINT_NAMES[3:] + SDK_JOINT_NAMES[:3]
        mapping = JointMap(names)
        sdk = np.arange(12, dtype=np.float32) + 0.25
        expected_policy = np.asarray([sdk[SDK_JOINT_NAMES.index(name)] for name in names])
        np.testing.assert_array_equal(mapping.to_policy(sdk), expected_policy)
        np.testing.assert_array_equal(mapping.to_sdk(expected_policy), sdk)
        for values in (sdk, sdk * 2, sdk * 3, sdk * 4, sdk * 5):
            np.testing.assert_array_equal(mapping.to_sdk(mapping.to_policy(values)), values)
        nominal = default_positions(names)
        for index, name in enumerate(names):
            self.assertAlmostEqual(float(nominal[index]), DEFAULT_ANGLES[name], places=6)
        np.testing.assert_array_equal(mapping.to_sdk(nominal), default_positions(SDK_JOINT_NAMES))

    def test_missing_duplicate_unknown_names_rejected(self):
        bad_cases = [None, SDK_JOINT_NAMES[:-1], [SDK_JOINT_NAMES[0]] * 12,
                     list(SDK_JOINT_NAMES[:-1]) + ["unknown_joint"]]
        for names in bad_cases:
            with self.subTest(names=names), self.assertRaises(ValueError):
                JointMap(names)

    def test_map_rejects_bad_shape_nonfinite_and_returns_copy(self):
        mapping = JointMap(SDK_JOINT_NAMES)
        for value in ([0] * 11, np.zeros((1, 12)), [float("nan")] * 12):
            with self.assertRaises(ValueError):
                mapping.to_policy(value)
        source = np.zeros(12, dtype=np.float32)
        result = mapping.to_policy(source)
        result[0] = 5
        self.assertEqual(source[0], 0)

    def test_target_scale_same_on_hips_and_others(self):
        q0 = default_positions(SDK_JOINT_NAMES)
        action = np.linspace(-1, 1, 12, dtype=np.float32)
        np.testing.assert_allclose(action_to_target(action, q0) - q0, action * 0.25, atol=1e-7)
        # Lab JointPositionAction clips the absolute processed target, not
        # raw normalized actions. These two boundaries are deliberately distinct.
        raw = np.asarray([-500, 500] + [0] * 10, dtype=np.float32)
        target = action_to_target(raw, q0)
        np.testing.assert_array_equal(target[:2], [-TARGET_CLIP, TARGET_CLIP])
        raw[0] = 101.0
        self.assertAlmostEqual(action_to_target(raw, q0)[0], q0[0] + .25 * 101., places=5)
        self.assertEqual(ACTION_SCALE, 0.25)
        lower, upper = joint_limits(SDK_JOINT_NAMES)
        self.assertTrue(np.all(q0 > lower))
        self.assertTrue(np.all(q0 < upper))


class ObservationTests(unittest.TestCase):
    def test_every_slice_has_source_units_and_order(self):
        q0 = default_positions(TRAINING_JOINT_NAMES)
        residual = np.arange(12, dtype=np.float32) / 10
        velocities = np.arange(12, dtype=np.float32) + 1
        previous = -np.arange(12, dtype=np.float32) / 3
        obs = build_observation([0.5, -0.25, 0.8], [1, -2, 3], [0, 0, -1],
                                q0 + residual, velocities, previous, q0)
        self.assertEqual(obs.shape, (45,))
        self.assertEqual(obs.dtype, np.float32)
        np.testing.assert_allclose(obs[0:3], [0.5, -0.25, 0.8])
        np.testing.assert_allclose(obs[3:6], [0.25, -0.5, 0.75])
        np.testing.assert_array_equal(obs[6:9], [0, 0, -1])
        np.testing.assert_allclose(obs[9:21], residual, atol=1e-7)
        np.testing.assert_allclose(obs[21:33], velocities * 0.05)
        np.testing.assert_array_equal(obs[33:45], previous)

    def test_clip_raw_terms_before_scaling_and_reject_invalid(self):
        zeros = np.zeros(12, dtype=np.float32)
        args = ([100, -100, 800], [800, -800, 0], [0, 0, -1],
                zeros, zeros, zeros, zeros)
        obs = build_observation(*args)
        np.testing.assert_array_equal(obs[:6], [100, -100, 100, 25, -25, 0])
        args = list(args)
        args[4] = np.asarray([1000, -1000] + [0] * 10)
        args[5] = np.asarray([250, -250] + [0] * 10)
        clipped = build_observation(*args)
        np.testing.assert_array_equal(clipped[21:23], [5, -5])
        np.testing.assert_array_equal(clipped[33:35], [100, -100])
        for index, value in [(0, [float("nan"), 0, 0]), (2, [0, 0, -9.81]),
                             (5, np.full(12, np.inf)), (3, np.zeros(11))]:
            bad = list(args)
            bad[index] = value
            with self.subTest(index=index), self.assertRaises(ValueError):
                build_observation(*bad)

    def test_history_is_newest_first_and_getters_never_advance(self):
        history = ObservationHistory()
        self.assertEqual(history.count, 0)
        self.assertFalse(history.ready)
        for label in range(1, 8):
            history.push(np.full(45, label, dtype=np.float32))
        self.assertEqual(history.count, 7)
        self.assertTrue(history.ready)
        result = history.get().reshape(6, 45)
        np.testing.assert_array_equal(result[:, 0], [7, 6, 5, 4, 3, 2])
        for _ in range(10):
            np.testing.assert_array_equal(history.get(), result.reshape(1, 270))
        self.assertEqual(history.count, 7)
        result.fill(99)
        self.assertEqual(history.get()[0, 0], 7)
        history.reset()
        self.assertEqual(history.count, 0)
        np.testing.assert_array_equal(history.get(), np.zeros((1, 270), dtype=np.float32))

    def test_history_partial_fill_zero_padding_and_failure_atomicity(self):
        history = ObservationHistory()
        frame = np.full(45, 3, dtype=np.float32)
        history.push(frame)
        frame.fill(9)
        self.assertEqual(history.get()[0, 0], 3)
        np.testing.assert_array_equal(history.get()[0, 45:], np.zeros(225))
        for bad in (np.zeros(44), np.full(45, np.inf), np.full(45, 101)):
            with self.assertRaises(ValueError):
                history.push(bad)
        self.assertEqual(history.count, 1)


class RotationTests(unittest.TestCase):
    @staticmethod
    def quaternion_from_rpy(rpy):
        r, p, y = np.asarray(rpy) / 2
        cr, cp, cy = math.cos(r), math.cos(p), math.cos(y)
        sr, sp, sy = math.sin(r), math.sin(p), math.sin(y)
        return np.asarray([cr*cp*cy + sr*sp*sy, sr*cp*cy - cr*sp*sy,
                           cr*sp*cy + sr*cp*sy, cr*cp*sy - sr*sp*cy])

    def test_roll_pitch_yaw_signs_and_gravity(self):
        eye = np.eye(3)
        np.testing.assert_allclose(projected_gravity([0, 0, 0], "rpy", eye), [0, 0, -1])
        np.testing.assert_allclose(projected_gravity([math.pi/2, 0, 0], "rpy", eye), [0, -1, 0], atol=1e-6)
        np.testing.assert_allclose(projected_gravity([0, math.pi/2, 0], "rpy", eye), [1, 0, 0], atol=1e-6)
        np.testing.assert_allclose(rotation_rpy([0, 0, math.pi/2]).dot([1, 0, 0]), [0, 1, 0], atol=1e-6)
        r, p = 0.2, -0.4
        expected = [math.sin(p), -math.sin(r)*math.cos(p), -math.cos(r)*math.cos(p)]
        for yaw in (0, 0.8, -2.0):
            np.testing.assert_allclose(projected_gravity([r, p, yaw], "rpy", eye), expected, atol=1e-6)

    def test_combined_attitudes_quaternion_sign_and_mounting(self):
        rng = np.random.RandomState(1234)
        mounting = rotation_rpy([0.2, -0.3, 0.4])
        for rpy in rng.uniform(-2, 2, size=(100, 3)):
            quat = self.quaternion_from_rpy(rpy)
            r_ws = rotation_rpy(rpy)
            np.testing.assert_allclose(rotation_quaternion_wxyz(quat), r_ws, atol=1e-12)
            np.testing.assert_allclose(rotation_quaternion_wxyz(-quat), r_ws, atol=1e-12)
            expected = mounting.dot(r_ws.T.dot([0, 0, -1]))
            for method, attitude in (("rpy", rpy), ("quaternion_wxyz", quat)):
                np.testing.assert_allclose(projected_gravity(attitude, method, mounting), expected, atol=1e-6)

    def test_invalid_rotations_do_not_become_identity(self):
        for bad in ([0, 0, 0, 0], [float("nan"), 0, 0, 0], [2, 0, 0, 0]):
            with self.assertRaises(ValueError):
                rotation_quaternion_wxyz(bad)
        for bad in (np.diag([1, 1, -1]), np.ones((3, 3)), np.eye(2)):
            with self.assertRaises(ValueError):
                validate_rotation_matrix(bad)
        with self.assertRaises(ValueError):
            rotation_rpy([0, float("nan"), 0])
        with self.assertRaises(ValueError):
            projected_gravity([0, 0, 0], "xyzw", np.eye(3))


if __name__ == "__main__":
    unittest.main()
