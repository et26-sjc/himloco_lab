import copy
import json
from pathlib import Path
import unittest

import numpy as np

from go1_lab_deploy.himloco.contract import SDK_JOINT_NAMES, TRAINING_JOINT_NAMES, default_positions
from fixtures import release_config
from go1_lab_deploy.himloco.runner import Controller, CommandPublisher, SafetyFault
from go1_lab_deploy.himloco.state import PROTOCOL_ID, StateError, StateReceiver
from go1_lab_deploy.lcm_types.leg_control_data_lcmt import leg_control_data_lcmt
from go1_lab_deploy.lcm_types.state_estimator_lcmt import state_estimator_lcmt
from go1_lab_deploy.lcm_types.rc_command_lcmt import rc_command_lcmt
from go1_lab_deploy.lcm_types.pd_tau_targets_lcmt import pd_tau_targets_lcmt


class FakeLCM:
    def __init__(self):
        self.callbacks, self.published = {}, []

    def subscribe(self, channel, callback):
        self.callbacks[channel] = callback
        return channel

    def unsubscribe(self, channel):
        del self.callbacks[channel]

    def publish(self, channel, data):
        self.published.append((channel, data))


def ready_config():
    # Read the actual release's inherited robot alignment/runtime evidence;
    # synthetic state samples below make no claim of new hardware validation.
    return release_config()


class StateTests(unittest.TestCase):
    def setUp(self):
        self.time = 1.0
        self.lc = FakeLCM()
        self.receiver = StateReceiver(self.lc, clock=lambda: self.time)
        self.rc()

    def rc(self, r2=False):
        m = rc_command_lcmt()
        m.right_lower_right_switch = int(r2)
        self.receiver._rc_cb("rc_command", m.encode())

    def pair(self, tick, marker=PROTOCOL_ID, stamp=None, order="il"):
        i, l = state_estimator_lcmt(), leg_control_data_lcmt()
        for m in (i, l):
            m.id, m.robot_id = tick, marker
            m.timestamp_us = stamp if stamp is not None else 1000000 + tick
        i.quat = [1., 0., 0., 0.]
        i.omegaBody = [0.3, -0.4, 0.5]
        l.q = default_positions(SDK_JOINT_NAMES).tolist()
        for ch in order:
            if ch == "i":
                self.receiver._imu_cb("state_estimator_data", i.encode())
            else:
                self.receiver._leg_cb("leg_control_data", l.encode())

    def test_pair_and_copy_and_raw_gyro(self):
        self.pair(10, order="i")
        with self.assertRaisesRegex(StateError, "Waiting"):
            self.receiver.snapshot()
        self.pair(10, order="l")
        s = self.receiver.snapshot()
        np.testing.assert_allclose(s["gyro_sensor"], [.3, -.4, .5])
        s["q_sdk"][:] = 99
        self.assertLess(self.receiver.snapshot()["q_sdk"].max(), 2)
        self.assertEqual(self.lc.published, [])

    def test_duplicates_do_not_refresh_age(self):
        self.pair(10)
        self.time += .06
        self.pair(10)
        self.rc()
        with self.assertRaisesRegex(StateError, "Stale SDK"):
            self.receiver.snapshot()

    def test_wrap_and_old_packets(self):
        self.pair(0xffffffff)
        self.time += .002
        self.pair(0)
        self.pair(0xfffffffe)
        self.assertEqual(self.receiver.snapshot()["source_tick"], 0)

    def test_protocol_and_fault_marker(self):
        for marker in (0, -PROTOCOL_ID):
            with self.subTest(marker=marker):
                r = StateReceiver(FakeLCM(), clock=lambda: self.time)
                m = state_estimator_lcmt()
                m.robot_id = marker
                r._imu_cb("", m.encode())
                with self.assertRaisesRegex(StateError, "protocol"):
                    r.snapshot()

    def test_r2_rising_edge_and_stale_rc(self):
        self.pair(1)
        self.rc(True)
        self.rc(True)
        self.assertEqual(self.receiver.snapshot()["r2_edges"], 1)
        self.rc(False)
        self.rc(True)
        self.assertEqual(self.receiver.snapshot()["r2_edges"], 2)
        self.time += .11
        self.pair(2)
        with self.assertRaisesRegex(StateError, "Stale RC"):
            self.receiver.snapshot()

    def test_malformed_is_latched(self):
        self.receiver._leg_cb("", b"invalid")
        self.pair(1)
        with self.assertRaises(StateError):
            self.receiver.snapshot()


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.cfg = ready_config()
        self.calls = []
        self.action = np.full(12, .1, dtype=np.float32)
        self.ctrl = Controller(self.cfg, self.policy)
        self.t, self.tick, self.edges = 0., 0, 0
        self.q = default_positions(SDK_JOINT_NAMES)

    def policy(self, history):
        self.calls.append(history.copy())
        return self.action.copy()

    def step(self, **changes):
        self.t += .02
        self.tick += 1
        s = dict(l2=False, state_age_s=.001, rc_age_s=.001,
                 q_sdk=self.q.copy(), dq_sdk=np.zeros(12),
                 gyro_sensor=np.array([.1, .2, .3]), rpy=np.zeros(3),
                 quaternion_wxyz=np.array([1., 0., 0., 0.]),
                 source_tick=self.tick, source_timestamp_us=1000000 + self.tick * 20000,
                 left_stick=np.zeros(2), right_stick=np.zeros(2), r2_edges=self.edges)
        s.update(changes)
        result = self.ctrl.step(s, self.t)
        if result["target_sdk"] is not None:
            self.q = result["target_sdk"].copy()
        return result

    def ready(self):
        self.assertIsNone(self.step()["target_sdk"])
        self.edges = 1
        self.step()
        for _ in range(200):
            r = self.step()
            if r["phase"] == "READY":
                break
        self.assertEqual(r["phase"], "READY")
        self.assertEqual(len(self.calls), 0)
        self.assertEqual(self.ctrl.history.count, 6)

    def test_full_handover_and_previous_applied_action(self):
        self.ready()
        self.edges = 2
        r0 = self.step()
        self.assertEqual(r0["ramp_alpha"], 0)
        self.assertEqual(len(self.calls), 1)
        r1 = self.step()
        np.testing.assert_allclose(r1["observation"][33:45], r0["applied_action"])
        for _ in range(55):
            previous = r1
            r1 = self.step()
            feedback = previous["applied_action"] if previous["ramp_alpha"] < 1. else previous["raw_action"]
            np.testing.assert_allclose(r1["observation"][33:45], feedback, atol=1e-7)
        self.assertEqual(r1["phase"], "RUNNING")
        np.testing.assert_allclose(r1["target_policy"], self.ctrl.q0 + .25 * self.action)
        r2 = self.step(left_stick=np.array([.5, 1.]), right_stick=np.array([.4, 0]))
        expected_command = (np.asarray([1., .5, .4]) * np.asarray(self.cfg["runtime"]["command_axis_signs"])
                            * np.asarray(self.cfg["runtime"]["max_abs_command"]))
        np.testing.assert_allclose(r2["commands"], expected_command)
        np.testing.assert_allclose(r2["observation"][:3], expected_command)
        np.testing.assert_array_equal(self.ctrl.previous_action, r2["raw_action"])
        self.assertEqual(r2["action_feedback_kind"], "policy_raw")
        # Retrieving/logging history never advances it.
        count = self.ctrl.history.count
        self.ctrl.history.get()
        self.ctrl.history.get()
        self.assertEqual(self.ctrl.history.count, count)

    def test_sdk_sensor_values_become_lab_native_policy_slices(self):
        self.ready()
        self.edges = 2
        for _ in range(55):
            previous = self.step()
        residual_sdk = np.arange(12, dtype=np.float32) * .005
        dq_sdk = np.arange(12, dtype=np.float32) + 1
        q_sdk = default_positions(SDK_JOINT_NAMES) + residual_sdk
        result = self.step(q_sdk=q_sdk, dq_sdk=dq_sdk)
        order = [SDK_JOINT_NAMES.index(n) for n in TRAINING_JOINT_NAMES]
        np.testing.assert_allclose(result["observation"][9:21], residual_sdk[order], atol=1e-7)
        np.testing.assert_allclose(result["observation"][21:33], dq_sdk[order] * .05)
        np.testing.assert_allclose(result["observation"][3:6], [.025, .05, .075])
        np.testing.assert_array_equal(result["observation"][33:45], previous["raw_action"])
        np.testing.assert_array_equal(self.calls[-1], result["history"])

    def test_folded_start_has_legal_targets_and_zero_initial_gains(self):
        self.q[0::3] = [-.86, .86, -.86, .86]
        self.q[2::3] = -2.80
        self.assertIsNone(self.step()["target_sdk"])
        self.edges = 1
        first = self.step()
        self.assertEqual(first["gain_scale"], 0.0)
        self.assertTrue(np.all(first["target_policy"] >= self.ctrl.lower))
        self.assertTrue(np.all(first["target_policy"] <= self.ctrl.upper))
        lc = FakeLCM()
        CommandPublisher(lc).publish(first)
        wire = pd_tau_targets_lcmt.decode(lc.published[-1][1])
        self.assertEqual(set(wire.kp + wire.kd), {0.0})
        previous_gain = 0.0
        for _ in range(200):
            result = self.step()
            self.assertGreaterEqual(result["gain_scale"], previous_gain)
            self.assertLessEqual(result["gain_scale"], 1.0)
            previous_gain = result["gain_scale"]
            if result["phase"] == "READY":
                break
        self.assertEqual(result["phase"], "READY")
        np.testing.assert_allclose(result["target_policy"], self.ctrl.q0)

    def test_tolerated_minus_2_70_start_no_longer_rejects_first_target(self):
        self.q[2::3] = -2.70
        self.step()
        self.edges = 1
        result = self.step()
        self.assertEqual(result["phase"], "CALIBRATE")
        self.assertTrue(np.all(result["target_policy"] >= self.ctrl.lower))

    def test_startup_allowance_is_bounded_and_not_used_when_running(self):
        self.q[2::3] = -3.0
        with self.assertRaisesRegex(SafetyFault, "startup"):
            self.step()
        self.ctrl = Controller(self.cfg, self.policy)
        self.q = default_positions(SDK_JOINT_NAMES)
        self.ready()
        self.edges = 2
        for _ in range(55):
            self.step()
        self.assertEqual(self.ctrl.phase, "RUNNING")
        self.q[2::3] = -2.73
        with self.assertRaisesRegex(SafetyFault, "Go1 URDF"):
            self.step()

    def test_r2_after_calibration_start_latches_stop(self):
        self.step()
        self.edges = 1
        self.step()
        self.edges = 2
        with self.assertRaisesRegex(SafetyFault, "R2 stop"):
            self.step()
        self.assertEqual(self.ctrl.phase, "STOPPED")
        with self.assertRaisesRegex(SafetyFault, "latched"):
            self.step()

    def test_multiple_r2_edges_cannot_start_calibration(self):
        self.step()
        self.edges = 2
        with self.assertRaisesRegex(SafetyFault, "Ambiguous R2"):
            self.step()
        self.assertIsNone(self.ctrl.last_target)

    def test_l2_stale_tilt_and_clock_faults(self):
        cases = [dict(l2=True), dict(state_age_s=.051),
                 dict(rc_age_s=.101), dict(rpy=np.array([1., 0., 0.]),
                                         quaternion_wxyz=np.array([np.cos(.5), np.sin(.5), 0., 0.]))]
        for changes in cases:
            with self.subTest(changes=changes):
                self.ctrl = Controller(self.cfg, self.policy)
                with self.assertRaises(SafetyFault):
                    self.step(**changes)
        self.ctrl = Controller(self.cfg, self.policy)
        self.step()
        self.t += .1
        with self.assertRaisesRegex(SafetyFault, "cycle gap"):
            self.step()

    def test_invalid_policy_cannot_produce_command(self):
        self.ready()
        self.edges = 2
        self.action[:] = np.nan
        with self.assertRaises(ValueError):
            self.step()
        self.assertEqual(self.ctrl.phase, "STOPPED")

    def test_limit_fault_instead_of_silent_policy_clamping(self):
        self.ready()
        self.edges = 2
        self.step()
        self.action[:] = 100
        with self.assertRaises(SafetyFault):
            for _ in range(20):
                self.step()

    def test_duplicate_source_cannot_create_history(self):
        self.ready()
        count = self.ctrl.history.count
        with self.assertRaisesRegex(SafetyFault, "No new SDK"):
            self.step(source_tick=self.tick)
        self.assertEqual(self.ctrl.history.count, count)

    def test_echoed_timestamp_and_stop_wire_protocol(self):
        self.ready()
        r = self.step()
        lc = FakeLCM()
        pub = CommandPublisher(lc)
        pub.publish(r)
        msg = pd_tau_targets_lcmt.decode(lc.published[-1][1])
        self.assertEqual(msg.robot_id, PROTOCOL_ID)
        self.assertEqual(msg.timestamp_us, r["source_timestamp_us"])
        np.testing.assert_allclose(msg.q_des, r["target_sdk"])
        self.assertEqual(set(msg.kp), {40.})
        self.assertEqual(set(msg.kd), {1.})
        self.assertEqual(set(msg.qd_des + msg.tau_ff), {0.})
        pub.stop()
        self.assertEqual(pd_tau_targets_lcmt.decode(lc.published[-1][1]).id, -1)

    def test_complete_runner_with_non_self_inverse_joint_order(self):
        names = list(TRAINING_JOINT_NAMES)
        self.assertEqual(self.cfg["model"]["training_dof_names"], names)
        self.assertNotEqual(names, list(SDK_JOINT_NAMES))
        self.ctrl = Controller(self.cfg, self.policy)
        self.action = np.arange(1, 13, dtype=np.float32) * .01
        self.ready()
        self.edges = 2
        for _ in range(60):
            result = self.step()
        self.assertEqual(result["phase"], "RUNNING")
        for sdk_index, name in enumerate(SDK_JOINT_NAMES):
            expected = default_positions(SDK_JOINT_NAMES)[sdk_index] + .25 * self.action[names.index(name)]
            self.assertAlmostEqual(result["target_sdk"][sdk_index], expected, places=6)

    def running(self):
        self.ready()
        self.edges = 2
        for _ in range(55):
            result = self.step()
        self.assertEqual(result["phase"], "RUNNING")

    def tracking_error(self, magnitude, joint=8):
        # Synthetic SDK feedback stays inside the mechanical envelope; only
        # lag behind the previously accepted target is varied in this test.
        measured = self.ctrl.last_target.copy()
        measured[joint] -= magnitude
        return self.step(q_sdk=self.ctrl.mapping.to_sdk(measured))

    def test_policy_short_tracking_lag_is_logged_and_recovers(self):
        self.running()
        for _ in range(5):
            result = self.tracking_error(.55)
            self.assertEqual(result["phase"], "RUNNING")
        self.assertAlmostEqual(result["tracking_error_over_limit_s"], .08)
        self.assertEqual(result["tracking_error_over_limit_joint"], "FL_calf_joint")
        result = self.tracking_error(.49)
        self.assertEqual(result["tracking_error_over_limit_s"], 0.)
        self.assertIsNone(result["tracking_error_over_limit_joint"])
        for _ in range(3):
            result = self.tracking_error(.55)
        self.assertAlmostEqual(result["tracking_error_over_limit_s"], .04)

    def test_same_joint_sustained_tracking_error_latches_stop(self):
        self.running()
        for _ in range(8):
            result = self.tracking_error(.55)
        self.assertLess(result["tracking_error_over_limit_s"], .15)
        previous = self.ctrl.last_target.copy()
        with self.assertRaises(SafetyFault) as caught:
            self.tracking_error(.55)
        details = caught.exception.details
        self.assertEqual(details["trigger"], "sustained")
        self.assertEqual(details["joint"], "FL_calf_joint")
        self.assertGreaterEqual(details["duration_s"], .15)
        self.assertFalse(details["candidate_sent"])
        self.assertEqual(self.ctrl.phase, "STOPPED")
        np.testing.assert_array_equal(self.ctrl.last_target, previous)

    def test_different_joints_cannot_accumulate_tracking_time(self):
        self.running()
        for i in range(24):
            result = self.tracking_error(.55, joint=8 + i % 2)
            self.assertEqual(result["tracking_error_over_limit_s"], 0.)
        self.assertEqual(result["phase"], "RUNNING")

    def test_large_tracking_error_stops_immediately_during_policy(self):
        self.running()
        with self.assertRaises(SafetyFault) as caught:
            self.tracking_error(1.01)
        self.assertEqual(caught.exception.details["trigger"], "instant")
        self.assertEqual(caught.exception.details["duration_s"], 0.)

    def test_ramp_uses_policy_persistence_without_changing_target_blending(self):
        self.ready()
        self.edges = 2
        self.step()
        for _ in range(4):
            result = self.tracking_error(.55)
            self.assertEqual(result["phase"], "RAMP")
            self.assertEqual(result["tracking_error_timeout_s"], .15)
            np.testing.assert_allclose(result["target_policy"],
                self.ctrl.q0 + .25 * result["applied_action"], atol=1e-7)

    def test_ready_tracking_error_keeps_original_immediate_threshold(self):
        self.ready()
        with self.assertRaises(SafetyFault) as caught:
            self.tracking_error(.55)
        self.assertEqual(caught.exception.details["phase"], "READY")
        self.assertEqual(caught.exception.details["timeout_s"], 0.)
        self.assertEqual(caught.exception.details["instant_limit_rad"], .5)

    def test_legacy_configs_without_optional_fields_keep_immediate_stop(self):
        self.cfg["runtime"].pop("tracking_error_timeout_s")
        self.cfg["runtime"].pop("max_instant_tracking_error_rad")
        self.ctrl = Controller(self.cfg, self.policy)
        self.running()
        with self.assertRaises(SafetyFault) as caught:
            self.tracking_error(.55)
        self.assertEqual(caught.exception.details["timeout_s"], 0.)


if __name__ == "__main__":
    unittest.main()
