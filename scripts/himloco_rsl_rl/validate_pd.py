"""Audit real 200 Hz explicit PD calls; no training checkpoint is loaded.

The flat, noise/randomization-free scene is for actuator diagnostics only.
Training/Play comparisons are also checked before these diagnostic overrides.
"""
import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--num_envs', type=int, default=32)
parser.add_argument('--output', default='logs/go1_pd_validation/control')
AppLauncher.add_app_launcher_args(parser)
parser.add_argument('--headless', action='store_true')
args = parser.parse_args()
launcher = AppLauncher(args)

import csv
import json
import unittest
from unittest.mock import patch

import gymnasium as gym
import torch
import warp as wp
import himloco_lab.tasks  # noqa: F401
from himloco_lab.rsl_rl import HimlocoVecEnvWrapper
from himloco_lab.tasks.locomotion.robots.go1.pd_env_cfg import Go1PDEnvCfg
from himloco_lab.tasks.locomotion.robots.go1.velocity_env_cfg import GO1_JOINT_NAMES


def main():
    torch.set_num_threads(4)
    suite = unittest.defaultTestLoader.discover(str(Path(__file__).resolve().parents[2] / 'tests'),
                                               pattern='test_pd_baseline.py')
    if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():
        raise AssertionError('PD configuration/formula regression failed')
    cfg = Go1PDEnvCfg()
    cfg.seed = 1
    cfg.scene.num_envs = args.num_envs
    cfg.scene.terrain.terrain_type = 'plane'
    cfg.scene.terrain.terrain_generator = None
    cfg.scene.terrain.max_init_terrain_level = None
    cfg.curriculum.terrain_levels = None
    cfg.curriculum.lin_vel_cmd_levels = None
    cfg.observations.policy.enable_corruption = False
    for name in ('physics_material', 'add_base_mass', 'randomize_rigid_body_com', 'external_force', 'push_robot'):
        setattr(cfg.events, name, None)
    cfg.events.reset_robot_joints.params['position_range'] = (1.0, 1.0)
    cfg.commands.base_velocity.heading_command = False
    cfg.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
    cfg.commands.base_velocity.low_vel_env_lin_x_ranges = (0.0, 0.0)
    cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
    env = HimlocoVecEnvWrapper(gym.make('Unitree-Go1-Velocity-PD', cfg=cfg), history_length=5)
    base = env.unwrapped
    robot = base.scene['robot']
    actuator = robot.actuators['base_legs']
    assert robot.joint_names == GO1_JOINT_NAMES
    assert type(actuator).__name__ == 'IdealPDActuator'
    assert not actuator.is_implicit_model
    assert not cfg.sim.use_newton_actuators and not base._physics_handles_decimation
    assert type(cfg.sim.physics).__name__ == 'PhysxCfg'
    assert (base.physics_dt, base.step_dt, cfg.decimation) == (0.005, 0.02, 4)
    torch.testing.assert_close(actuator.stiffness, torch.full_like(actuator.stiffness, 40.0))
    torch.testing.assert_close(actuator.damping, torch.ones_like(actuator.damping))
    # These are the actual backend drive gains, independent of the explicit gains.
    torch.testing.assert_close(robot.data.joint_stiffness.torch, torch.zeros_like(robot.data.joint_stiffness.torch))
    torch.testing.assert_close(robot.data.joint_damping.torch, torch.zeros_like(robot.data.joint_damping.torch))
    assert torch.count_nonzero(wp.to_torch(robot.root_view.get_dof_stiffnesses())) == 0
    assert torch.count_nonzero(wp.to_torch(robot.root_view.get_dof_dampings())) == 0
    expected_limits = torch.tensor([23.7] * 8 + [35.55] * 4, device=env.device)
    torch.testing.assert_close(actuator.actuator_effort_limit[0], expected_limits)
    records = []
    errors = []
    heights = []
    total_done = 0
    first_zero_torque = None
    original_compute = actuator.compute
    current_case = 'zero'
    policy_step = 0
    calls = []
    recording = True

    def audited_compute(control_action, joint_pos, joint_vel):
        nonlocal first_zero_torque
        if not recording:
            return original_compute(control_action, joint_pos, joint_vel)
        target = control_action.joint_positions.clone()
        q, dq = joint_pos.clone(), joint_vel.clone()
        torch.testing.assert_close(control_action.joint_velocities, torch.zeros_like(dq))
        torch.testing.assert_close(control_action.joint_efforts, torch.zeros_like(dq))
        theory = 40.0 * (target - q) - dq
        result = original_compute(control_action, joint_pos, joint_vel)
        clipped = theory.clamp(-actuator.actuator_effort_limit, actuator.actuator_effort_limit)
        torch.testing.assert_close(actuator.computed_effort, theory, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(actuator.applied_effort, clipped, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(result.joint_efforts, clipped)
        if first_zero_torque is None:
            first_zero_torque = actuator.computed_effort.abs().max().item()
        errors.append((actuator.computed_effort - theory).abs().max().item())
        substep = len(calls)
        calls.append((target.clone(), q.clone(), dq.clone(), actuator.computed_effort.clone(),
                      actuator.applied_effort.clone()))
        # Bounded diagnostic capture: all 12 joints of environment 0, not every env.
        values = torch.stack((target[0], q[0], dq[0], actuator.computed_effort[0],
                              actuator.applied_effort[0], theory[0]), dim=-1).cpu().tolist()
        for name, value in zip(robot.joint_names, values):
            records.append([current_case, policy_step, substep, base._sim_step_counter, name, *value])
        return result

    with patch.object(actuator, 'compute', audited_compute), torch.inference_mode():
        # Observe settling/standing under zero action before deliberately saturating.
        for policy_step in range(100):
            calls.clear()
            counter = base._sim_step_counter
            _, _, reward, done, _, _, _ = env.step(torch.zeros(env.num_envs, 12, device=env.device))
            assert len(calls) == 4 and base._sim_step_counter - counter == 4
            assert torch.isfinite(reward).all()
            total_done += int(done.sum())
            heights.append(robot.data.root_pos_w.torch[:, 2].clone())
        zero_action_calls = policy_step + 1
        assert first_zero_torque < 1e-4, first_zero_torque
        assert total_done == 0, f'Nominal zero-action stance terminated {total_done} times'
        # Check ordinary tracking and both signs of saturation during real stepping.
        for current_case, value in [('fixed', 0.4), ('positive_saturation', 20.0), ('negative_saturation', -20.0)]:
            recording = False  # reset may submit commands without advancing physics
            env.reset()
            recording = True
            calls.clear()
            policy_step += 1
            action = torch.full((env.num_envs, 12), value, device=env.device)
            target = robot.data.default_joint_pos.torch + 0.25 * action
            env.step(action)
            assert len(calls) == 4
            for actual_target, _, _, _, _ in calls:
                torch.testing.assert_close(actual_target, target)
            assert (calls[-1][1] - calls[0][1]).abs().max() > 1e-6, 'Stale q across substeps'
            assert (calls[-1][3] - calls[0][3]).abs().max() > 1e-6, 'Torque held across substeps'
            torch.testing.assert_close(robot.actuators.computed_effort.torch, calls[-1][3])
            torch.testing.assert_close(robot.actuators.applied_effort.torch, calls[-1][4])
            if 'saturation' in current_case:
                sign = 1 if value > 0 else -1
                torch.testing.assert_close(calls[0][4], sign * expected_limits.expand(env.num_envs, -1))
    height = torch.stack(heights)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'substeps.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['case', 'policy_step', 'substep', 'physics_step', 'joint', 'target_position_rad',
                         'position_rad', 'velocity_rad_s', 'computed_torque_nm', 'applied_torque_nm', 'theory_torque_nm'])
        writer.writerows(records)
    result = dict(status='passed', unit_tests=suite.countTestCases(), actuator=type(actuator).__name__,
                  num_envs=env.num_envs, joint_names=robot.joint_names, physics_dt=base.physics_dt,
                  policy_dt=base.step_dt, compute_calls_per_policy_step=4, audited_compute_calls=len(errors),
                  max_formula_error=max(errors), zero_action_first_torque_max=first_zero_torque,
                  zero_action_duration_s=zero_action_calls * base.step_dt, zero_action_terminations=total_done,
                  zero_action_root_height_min=height.min().item(), zero_action_root_height_max=height.max().item(),
                  zero_action_final_root_height_mean=height[-1].mean().item(),
                  default_joint_pos=robot.data.default_joint_pos.torch[0].tolist(),
                  stiffness=actuator.stiffness[0].tolist(), damping=actuator.damping[0].tolist(),
                  model_effort_limits=actuator.actuator_effort_limit[0].tolist(),
                  model_velocity_limits=actuator.actuator_velocity_limit[0].tolist(),
                  solver_effort_limits=robot.data.joint_effort_limits.torch[0].tolist(),
                  solver_velocity_limits=robot.data.joint_vel_limits.torch[0].tolist(),
                  solver_stiffness=robot.data.joint_stiffness.torch[0].tolist(),
                  solver_damping=robot.data.joint_damping.torch[0].tolist(),
                  backend_drive_gains_zero=True,
                  capture='Environment 0, all joints; command torque, not measured contact/reaction torque',
                  diagnostic_overrides='plane, no noise/material/mass/COM/push/wrench randomization; nominal reset joints')
    (output / 'validation.json').write_text(json.dumps(result, indent=2) + '\n')
    print('PD_CONTROL_VALIDATION_OK', json.dumps(result))
    env.close()


try:
    main()
except BaseException:
    import traceback
    traceback.print_exc()
    launcher.app.close(exit_code=1)
else:
    launcher.app.close()
