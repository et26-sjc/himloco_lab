"""PD changes only the actuator; run after AppLauncher via validate_pd.py."""
import unittest

import gymnasium as gym
import torch
from isaaclab.actuators import IdealPDActuator
from isaaclab.utils.types import ArticulationActions

import himloco_lab.tasks  # noqa: F401
from himloco_lab.tasks.locomotion.agents.himloco_rsl_rl_cfg import Go1PPORunnerCfg, Go1PDPPORunnerCfg
from himloco_lab.tasks.locomotion.robots.go1.velocity_env_cfg import Go1EnvCfg, Go1PlayEnvCfg, GO1_JOINT_NAMES
from himloco_lab.tasks.locomotion.robots.go1.pd_env_cfg import Go1PDEnvCfg, Go1PDPlayEnvCfg, GO1_PD_ACTUATOR_CFG


class PDBaselineContracts(unittest.TestCase):
    def test_only_actuator_differs(self):
        for normal, pd in ((Go1EnvCfg, Go1PDEnvCfg), (Go1PlayEnvCfg, Go1PDPlayEnvCfg)):
            left, right = normal().to_dict(), pd().to_dict()
            left['scene']['robot'].pop('actuators')
            right['scene']['robot'].pop('actuators')
            self.assertEqual(left, right)

    def test_runner_changes_only_experiment_name(self):
        left, right = Go1PPORunnerCfg().to_dict(), Go1PDPPORunnerCfg().to_dict()
        self.assertEqual(left.pop('experiment_name'), 'go1_rough')
        self.assertEqual(right.pop('experiment_name'), 'go1_rough_pd')
        self.assertEqual(left, right)
        self.assertFalse(right['resume'])
        self.assertEqual(right['max_iterations'], 10000)

    def test_registration_and_config_isolation(self):
        for suffix in ('', '-Play'):
            self.assertIn('pd_env_cfg:', gym.spec(f'Unitree-Go1-Velocity-PD{suffix}').kwargs['env_cfg_entry_point'])
            self.assertIn('.velocity_env_cfg:', gym.spec(f'Unitree-Go1-Velocity{suffix}').kwargs['env_cfg_entry_point'])
        pd = Go1PDEnvCfg()
        pd.scene.robot.actuators['base_legs'].stiffness = 123.0
        self.assertEqual(Go1PDEnvCfg().scene.robot.actuators['base_legs'].stiffness, 40.0)
        self.assertEqual(Go1EnvCfg().scene.robot.actuators['base_legs'].class_type.__name__, 'ActuatorNetMLP')

    def test_pd_formula_and_saturation(self):
        actuator = IdealPDActuator(GO1_PD_ACTUATOR_CFG.copy(), GO1_JOINT_NAMES, slice(None), 2, 'cpu')
        torch.manual_seed(17)
        q, dq = torch.randn(2, 12), torch.randn(2, 12)
        target = q + torch.tensor([0.02, -0.02, 20., -20.] * 3)
        command = ArticulationActions(joint_positions=target, joint_velocities=torch.zeros_like(q),
                                      joint_efforts=torch.zeros_like(q))
        actual = actuator.compute(command, q, dq)
        expected = 40.0 * (target - q) - dq
        limits = torch.tensor([23.7] * 8 + [35.55] * 4).expand_as(q)
        torch.testing.assert_close(actuator.computed_effort, expected)
        torch.testing.assert_close(actual.joint_efforts, expected.clamp(-limits, limits))
        self.assertIsNone(actual.joint_positions)
        self.assertIsNone(actual.joint_velocities)


if __name__ == '__main__':
    unittest.main()
