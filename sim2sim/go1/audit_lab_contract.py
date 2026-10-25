"""Optional development audit: transplant live Lab states into MuJoCo, compare obs.

This is the only simulator audit requiring Isaac Lab. The standalone runtime does
not import it. No training/update occurs; the already trained JIT controls Lab.
"""
import argparse
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--policy', required=True)
parser.add_argument('--output', default='artifacts/sim2sim/lab_contract')
parser.add_argument('--steps', type=int, default=32)
parser.add_argument('--gait-diagnostics', action='store_true',
                    help='Also record steady gait and same-pose knee/foot forward kinematics')
AppLauncher.add_app_launcher_args(parser)
parser.add_argument('--headless', action='store_true')
args = parser.parse_args()
launcher = AppLauncher(args)

import json
import unittest
import gymnasium as gym
import mujoco
import numpy as np
import torch
import warp as wp
import himloco_lab.tasks
from himloco_lab.rsl_rl import HimlocoVecEnvWrapper
from himloco_lab.tasks.locomotion.robots.go1.pd_env_cfg import Go1PDPlayEnvCfg
from sim2sim.go1.config import SimConfig
from sim2sim.go1.mujoco_adapter import MujocoAdapter
from sim2sim.go1.policy_runner import HistoryBuffer, PolicyRunner


def main():
    torch.set_num_threads(1)
    suite = unittest.defaultTestLoader.discover(str(Path(__file__).resolve().parents[2]/'tests'),
                                               pattern='test_him_contracts.py')
    if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():
        raise AssertionError('Existing HIM regression failed')
    cfg = Go1PDPlayEnvCfg()
    cfg.seed = 1
    cfg.scene.num_envs = 4
    cfg.scene.terrain.terrain_type = 'plane'
    cfg.scene.terrain.terrain_generator = None
    if args.gait_diagnostics:
        cfg.episode_length_s = args.steps * cfg.sim.dt * cfg.decimation + 1.
        cfg.events.physics_material = None
        cfg.events.reset_robot_joints.params['position_range'] = (1., 1.)
        cfg.events.reset_base.params['pose_range'] = {'x': (0., 0.), 'y': (0., 0.), 'yaw': (0., 0.)}
    env = HimlocoVecEnvWrapper(gym.make('Unitree-Go1-Velocity-PD-Play', cfg=cfg), history_length=5)
    robot = env.unwrapped.scene['robot']
    policy = PolicyRunner(args.policy)
    adapter = MujocoAdapter(SimConfig())
    histories = [HistoryBuffer() for _ in range(env.num_envs)]
    order = [robot.joint_names.index(n) for n in adapter.names]
    errors = np.zeros(3)
    frames = []
    gait = []
    active = np.ones(env.num_envs, dtype=bool)
    max_fk_error = np.zeros(2)
    legs = ('FL', 'FR', 'RL', 'RR')
    knee_ids = [robot.body_names.index(f'{leg}_calf') for leg in legs]
    foot_ids = [robot.body_names.index(f'{leg}_foot') for leg in legs]
    mj_knee_ids = [mujoco.mj_name2id(adapter.model, mujoco.mjtObj.mjOBJ_BODY, f'{leg}_calf') for leg in legs]
    for step in range(args.steps):
        lab_history = env.get_observations().cpu().numpy()
        command = env.unwrapped.command_manager.get_command('base_velocity').cpu().numpy()
        knees, feet, mj_knees, mj_feet = [], [], [], []
        for i in range(env.num_envs):
            qxyzw = robot.data.root_quat_w.torch[i].cpu().numpy()
            adapter.data.qpos[adapter.root_qpos:adapter.root_qpos+3] = robot.data.root_pos_w.torch[i].cpu().numpy()
            adapter.data.qpos[adapter.root_qpos+3:adapter.root_qpos+7] = qxyzw[[3, 0, 1, 2]]
            adapter.data.qpos[adapter.qpos_ids] = robot.data.joint_pos.torch[i, order].cpu().numpy()
            adapter.data.qvel[:] = 0
            adapter.data.qvel[adapter.root_dof+3:adapter.root_dof+6] = robot.data.root_ang_vel_b.torch[i].cpu().numpy()
            adapter.data.qvel[adapter.dof_ids] = robot.data.joint_vel.torch[i, order].cpu().numpy()
            mujoco.mj_forward(adapter.model, adapter.data)
            if args.gait_diagnostics:
                rotation = adapter.data.xmat[adapter.body_id].reshape(3, 3)
                root = robot.data.root_pos_w.torch[i].cpu().numpy()
                lab_bodies = robot.data.body_pos_w.torch[i].cpu().numpy()
                knee = (lab_bodies[knee_ids] - root) @ rotation
                foot = (lab_bodies[foot_ids] - root) @ rotation
                mj_knee = (adapter.data.xpos[mj_knee_ids] - adapter.data.xpos[adapter.body_id]) @ rotation
                mj_foot = (adapter.data.geom_xpos[adapter.foot_geom_ids] - adapter.data.xpos[adapter.body_id]) @ rotation
                max_fk_error = np.maximum(max_fk_error, [np.max(np.abs(knee-mj_knee)), np.max(np.abs(foot-mj_foot))])
                np.testing.assert_allclose(knee, mj_knee, atol=1e-4, rtol=1e-4)
                np.testing.assert_allclose(foot, mj_foot, atol=1e-4, rtol=1e-4)
                knees.append(knee); feet.append(foot); mj_knees.append(mj_knee); mj_feet.append(mj_foot)
            previous = env.unwrapped.action_manager.action[i].cpu().numpy()
            frame = adapter.observation(command[i], previous)
            history = histories[i].push(frame)
            np.testing.assert_allclose(frame, lab_history[i, :45], atol=2e-5, rtol=2e-5)
            np.testing.assert_allclose(history[0], lab_history[i], atol=2e-5, rtol=2e-5)
            with torch.no_grad():
                actual = policy.infer(history)
                expected = policy.infer(lab_history[i:i+1].copy())
            np.testing.assert_allclose(actual, expected, atol=3e-4, rtol=3e-4)
            errors = np.maximum(errors, [np.max(np.abs(frame-lab_history[i, :45])),
                                         np.max(np.abs(history[0]-lab_history[i])), np.max(np.abs(actual-expected))])
        frames.append(lab_history.copy())
        if args.gait_diagnostics:
            gait.append(dict(time=step*cfg.sim.dt*cfg.decimation, active=active.copy(),
                q=robot.data.joint_pos.torch[:, order].cpu().numpy().copy(),
                q_target=robot.actuators.target_command.position.torch[:, order].cpu().numpy().copy(),
                action=env.unwrapped.action_manager.action.cpu().numpy().copy(),
                knee_body=np.array(knees), foot_body=np.array(feet),
                mj_knee_body=np.array(mj_knees), mj_foot_body=np.array(mj_feet),
                base_position=robot.data.root_pos_w.torch.cpu().numpy().copy(),
                base_velocity=robot.data.root_lin_vel_b.torch.cpu().numpy().copy()))
        with torch.no_grad():
            actions = policy.model(torch.from_numpy(lab_history.copy())).to(env.device)
            _, _, _, _, _, reset_ids, _ = env.step(actions)
        for i in reset_ids.cpu().tolist():
            histories[i].reset()
            active[i] = False
    record = dict(status='passed', num_envs=4, steps=args.steps, him_unit_tests=suite.countTestCases(),
                  max_frame_error=errors[0], max_history_error=errors[1], max_jit_action_error=errors[2],
                  comparison='Identical transplanted poses/joint states, not trajectory equivalence',
                  lab_joint_names=robot.joint_names,
                  lab_joint_limits=robot.data.joint_pos_limits.torch[0].cpu().tolist(),
                  lab_solver_effort_limits=robot.data.joint_effort_limits.torch[0].cpu().tolist(),
                  lab_solver_velocity_limits=robot.data.joint_vel_limits.torch[0].cpu().tolist(),
                  lab_armature=robot.data.joint_armature.torch[0].cpu().tolist(),
                  lab_joint_friction=robot.data.joint_friction_coeff.torch[0].cpu().tolist(),
                  lab_total_mass=float(wp.to_torch(robot.root_view.get_masses())[0].sum()),
                  policy=policy.metadata)
    if args.gait_diagnostics:
        record['forward_kinematics_max_error_m'] = dict(knee=float(max_fk_error[0]), foot=float(max_fk_error[1]))
        record['gait_first_episode_survivors'] = int(active.sum())
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output/'validation.json').write_text(json.dumps(record, indent=2)+'\n')
    torch.save({'lab_observations':torch.from_numpy(np.concatenate(frames))}, output/'observations.pt')
    if gait:
        np.savez_compressed(output/'gait.npz', **{key:np.asarray([row[key] for row in gait]) for key in gait[0]})
    print('LAB_MUJOCO_CONTRACT_OK', json.dumps(record))
    env.close()


try:
    main()
except BaseException:
    import traceback
    traceback.print_exc()
    launcher.app.close(exit_code=1)
else:
    launcher.app.close()
