"""CPU-only MuJoCo control/observation tests; set HIM_GO1_TEST_POLICY for JIT tests."""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

import mujoco
import numpy as np

from sim2sim.go1.config import SimConfig, gain_vector, load_config, LAB_JOINT_NAMES
from sim2sim.go1.controllers import PDController
from sim2sim.go1.mujoco_adapter import MujocoAdapter
from sim2sim.go1.policy_runner import HistoryBuffer, PolicyRunner
from sim2sim.go1.simulation import run_episode


class ZeroPolicy:
    metadata = {'test_policy': 'zero_action'}
    def reset(self):
        self.calls = 0
    def act(self, frame):
        self.calls += 1
        return np.zeros(12)


class Contracts(unittest.TestCase):
    def test_named_joint_mapping_and_torque_model(self):
        a = MujocoAdapter(SimConfig())
        self.assertEqual(a.names, LAB_JOINT_NAMES)
        self.assertFalse(np.array_equal(a.qpos_ids, np.arange(7, 19)))
        for i, name in enumerate(a.names):
            self.assertEqual(a.qpos_ids[i], a.model.joint(name).qposadr[0])
            self.assertEqual(a.dof_ids[i], a.model.joint(name).dofadr[0])
        a.data.qpos[a.qpos_ids] += np.arange(12)*.01
        a.data.qvel[a.dof_ids] = np.arange(12)*.1
        mujoco.mj_forward(a.model, a.data)
        obs = a.observation([.7, .2, .3], np.arange(12))
        np.testing.assert_allclose(obs[:3], [.7, .2, .3])
        np.testing.assert_allclose(obs[9:21], np.arange(12)*.01, atol=1e-7)
        np.testing.assert_allclose(obs[21:33], np.arange(12)*.005, atol=1e-7)
        np.testing.assert_array_equal(obs[33:], np.arange(12))

    def test_reject_position_servos(self):
        cfg = SimConfig()
        cfg.model = str(Path(cfg.model).with_name('scene.xml'))
        with self.assertRaisesRegex(ValueError, 'no position drive'):
            MujocoAdapter(cfg)

    def test_gravity_and_body_angular_velocity(self):
        a = MujocoAdapter(SimConfig())
        np.testing.assert_allclose(a.state()['gravity'], [0, 0, -1], atol=1e-10)
        i = a.root_qpos
        a.data.qpos[i+3:i+7] = [np.sqrt(.5), 0, np.sqrt(.5), 0]
        mujoco.mj_forward(a.model, a.data)
        np.testing.assert_allclose(a.state()['gravity'], [1, 0, 0], atol=1e-10)
        a.reset()
        a.data.qpos[i+3:i+7] = [np.sqrt(.5), 0, 0, np.sqrt(.5)]
        a.data.qvel[3:6] = [1, 2, 3]  # free angular qvel is body aligned
        mujoco.mj_forward(a.model, a.data)
        np.testing.assert_allclose(a.state()['omega'], [1, 2, 3], atol=1e-10)
        np.testing.assert_allclose(a.state()['gravity'], [0, 0, -1], atol=1e-10)

    def test_no_privileged_velocity_and_clip_before_scale(self):
        a = MujocoAdapter(SimConfig())
        before = a.observation([1, 0, 0], np.zeros(12))
        a.data.qvel[:3] = [5, -2, 3]
        mujoco.mj_forward(a.model, a.data)
        np.testing.assert_array_equal(before, a.observation([1, 0, 0], np.zeros(12)))
        a.data.qvel[a.dof_ids] = 200
        a.data.qvel[3] = 200
        mujoco.mj_forward(a.model, a.data)
        frame = a.observation([1, 0, 0], np.zeros(12))
        self.assertEqual(frame[3], 25.)
        np.testing.assert_array_equal(frame[21:33], np.full(12, 5.))

    def test_history_order_reset_padding(self):
        h = HistoryBuffer()
        for i in range(1, 8):
            value = h.push(np.full(45, i))
        np.testing.assert_array_equal(value.reshape(6, 45)[:, 0], [7, 6, 5, 4, 3, 2])
        h.reset()
        value = h.push(np.full(45, 9))
        np.testing.assert_array_equal(value[0, :45], np.full(45, 9))
        self.assertEqual(np.count_nonzero(value[0, 45:]), 0)

    def test_cli_precedence_and_group_gains(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'config.yaml'
            path.write_text('controller:\n  kp: {hip: 20.5, thigh: 30, calf: 40}\n  kd: 0.5\n')
            cfg = load_config(path, kd=.75)
            values = gain_vector(cfg.controller.kp, cfg.joint_names)
            np.testing.assert_array_equal(values, [20.5]*4+[30.]*4+[40.]*4)
            self.assertEqual(cfg.controller.kd, .75)
            self.assertEqual(load_config(path, kp=31.25).controller.kp, 31.25)
        with self.assertRaises(ValueError):
            gain_vector({'not_a_joint': 1}, LAB_JOINT_NAMES)

    def test_passive_damping_override_is_separate_from_pd(self):
        original = MujocoAdapter(SimConfig())
        overridden = MujocoAdapter(SimConfig(passive_damping=0.))
        np.testing.assert_array_equal(overridden.model.dof_damping[overridden.dof_ids], np.zeros(12))
        np.testing.assert_array_equal(overridden.authored_passive_damping,
                                      original.model.dof_damping[original.dof_ids])
        np.testing.assert_array_equal(original.model.dof_armature, overridden.model.dof_armature)
        np.testing.assert_array_equal(original.model.dof_frictionloss, overridden.model.dof_frictionloss)
        controller = PDController(overridden.cfg.controller, overridden.names, overridden.effort_limits)
        np.testing.assert_array_equal(controller.kp, np.full(12, 40.))
        np.testing.assert_array_equal(controller.kd, np.ones(12))
        self.assertEqual(original.model_metadata()['xml_sha256'], overridden.model_metadata()['xml_sha256'])
        self.assertNotEqual(original.model_metadata()['effective_dynamics_sha256'],
                            overridden.model_metadata()['effective_dynamics_sha256'])
        np.testing.assert_array_equal(original.observation([.7,0,0], np.zeros(12)),
                                      overridden.observation([.7,0,0], np.zeros(12)))

    def test_pd_gains_clipping_and_physical_torque(self):
        a = MujocoAdapter(SimConfig())
        for kp, kd in [(40., 1.), (20., .5), (30., .75)]:
            cfg = copy.deepcopy(a.cfg.controller)
            cfg.kp, cfg.kd = kp, kd
            c = PDController(cfg, a.names, a.effort_limits)
            q, dq = a.joints()
            dq[:] = np.linspace(-2, 2, 12)
            target = q + np.array([.01, -.02, 10, -10]*3)
            torque = c.step(target, q, dq)
            np.testing.assert_allclose(c.computed_torque, kp*(target-q)-kd*dq)
            np.testing.assert_allclose(torque, np.clip(c.computed_torque, -a.effort_limits, a.effort_limits))
            a.data.ctrl[a.actuator_ids] = torque
            mujoco.mj_forward(a.model, a.data)
            np.testing.assert_allclose(a.data.actuator_force[a.actuator_ids], torque)
            np.testing.assert_allclose(a.data.qfrc_actuator[a.dof_ids], torque)

    def test_substep_frequency_and_live_pd(self):
        cfg = SimConfig()
        for kp, kd in [(40., 1.), (20., .5)]:
            cfg.controller.kp, cfg.controller.kd = kp, kd
            policy = ZeroPolicy()
            # Include ground contact: in early free fall nominal joints need no PD effort.
            result, trace = run_episode(cfg, policy, [0, 0, 0], .4)
            self.assertEqual(result['physics_steps'], 80)
            self.assertEqual(result['policy_calls'], 20)
            self.assertEqual(policy.calls, 20)
            np.testing.assert_allclose(np.diff(trace['time']), .005, atol=1e-12)
            np.testing.assert_allclose(trace['computed_torque'], kp*(trace['q_target']-trace['q'])-kd*trace['dq'])
            self.assertGreater(np.ptp(trace['computed_torque']), .01)

    def test_reset_and_legacy_profile_are_explicit(self):
        a = MujocoAdapter(SimConfig())
        initial = a.initial_state_sha256()
        a.step(np.ones(12))
        a.reset()
        self.assertEqual(a.initial_state_sha256(), initial)
        legacy = MujocoAdapter(SimConfig(profile='legacy'))
        np.testing.assert_allclose(legacy.observation([.7, .2, .3], np.zeros(12))[:3], [1.4, .4, .075])
        self.assertEqual(legacy.names[1], 'FL_thigh_joint')
        legacy.data.qvel[legacy.dof_ids] = 200
        mujoco.mj_forward(legacy.model, legacy.data)
        np.testing.assert_array_equal(legacy.observation([0, 0, 0], np.zeros(12))[21:33], np.full(12, 10.))
        np.testing.assert_allclose(legacy.targets(np.full(12, 200)), legacy.default_pos+25.)
        self.assertFalse(any(n.startswith(('isaaclab', 'isaacsim', 'himloco_lab')) for n in sys.modules))

    def test_jit_shapes_and_repeatability(self):
        path = os.environ.get('HIM_GO1_TEST_POLICY')
        if not path:
            self.skipTest('Set HIM_GO1_TEST_POLICY to a validated JIT with manifest')
        p = PolicyRunner(path)
        a = MujocoAdapter(SimConfig())
        frame = a.observation([.7, 0, .3], np.zeros(12))
        first = p.act(frame)
        p.reset()
        np.testing.assert_array_equal(first, p.act(frame))
        self.assertEqual(first.shape, (12,))
        self.assertTrue(np.isfinite(first).all())
        with self.assertRaises(ValueError):
            PolicyRunner(path, profile='legacy')


if __name__ == '__main__':
    unittest.main()
