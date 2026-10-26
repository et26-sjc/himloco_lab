"""Record native Lab sensor/observation/action pairs for offline deployment replay.

This host-only verification tool requires Isaac Lab; the deployment runtime does
not. It executes the already exported JIT without training, LCM or robot I/O.
Recorded samples are pre-action states, paired with that action's *actual* Lab
JointPositionAction processed target, never with a post-reset observation.
"""
import argparse
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import platform
import subprocess
import sys

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--policy', required=True, help='Combined PD Estimator+Actor TorchScript JIT')
parser.add_argument('--output', default='artifacts/go1_lab_deploy/lab_reference')
parser.add_argument('--steps', type=int, default=256)
parser.add_argument('--num_envs', type=int, default=4)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--velocity', type=float, nargs=3, default=(.7, 0., 0.))
AppLauncher.add_app_launcher_args(parser)
parser.add_argument('--headless', action='store_true')
args = parser.parse_args()
if args.steps < 1 or args.num_envs < 1:
    parser.error('--steps and --num_envs must be positive')
launcher = AppLauncher(args)
sys.argv = [sys.argv[0]]

import gymnasium as gym
import numpy as np
import torch

import himloco_lab.tasks  # noqa: F401 - task registration after SimulationApp
from himloco_lab.rsl_rl import HimlocoVecEnvWrapper
from himloco_lab.tasks.locomotion.robots.go1.pd_env_cfg import Go1PDPlayEnvCfg
from himloco_lab.tasks.locomotion.robots.go1.velocity_env_cfg import GO1_JOINT_NAMES
from isaaclab.envs.mdp.actions.joint_actions import JointAction


SDK_JOINT_NAMES = tuple(f'{leg}_{joint}_joint' for leg in ('FR', 'FL', 'RR', 'RL')
                        for joint in ('hip', 'thigh', 'calf'))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array(tensor):
    return tensor.detach().cpu().numpy().copy()


def version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def main():
    torch.set_num_threads(1)
    if not np.isfinite(args.velocity).all():
        raise ValueError('--velocity must contain finite physical-unit commands')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output/'reference.npz').exists() or (output/'metadata.json').exists():
        raise FileExistsError('Select a new output directory; existing reference data is preserved')
    policy_path = Path(args.policy).resolve()
    model = torch.jit.load(str(policy_path), map_location='cpu').eval()
    manifest_path = policy_path.with_suffix('.manifest.json')
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    policy_hash = sha256(policy_path)
    if manifest and (manifest.get('policy_sha256') != policy_hash or manifest.get('profile') != 'lab'):
        raise ValueError('JIT manifest hash/profile does not match the Lab policy')
    cfg = Go1PDPlayEnvCfg()
    cfg.seed = args.seed
    cfg.sim.device = args.device
    cfg.scene.num_envs = args.num_envs
    cfg.scene.terrain.terrain_type = 'plane'
    cfg.scene.terrain.terrain_generator = None
    cfg.episode_length_s = (args.steps + 1) * cfg.sim.dt * cfg.decimation + 1.
    cfg.observations.policy.enable_corruption = False
    cfg.curriculum.terrain_levels = None
    cfg.curriculum.lin_vel_cmd_levels = None
    for name in ('physics_material', 'add_base_mass', 'randomize_rigid_body_com', 'external_force', 'push_robot'):
        setattr(cfg.events, name, None)
    cfg.events.reset_robot_joints.params['position_range'] = (1., 1.)
    cfg.events.reset_robot_joints.params['velocity_range'] = (0., 0.)
    cfg.events.reset_base.params['pose_range'] = {k: (0., 0.) for k in ('x', 'y', 'yaw')}
    cfg.events.reset_base.params['velocity_range'] = {
        k: (0., 0.) for k in ('x', 'y', 'z', 'roll', 'pitch', 'yaw')}
    cfg.commands.base_velocity.resampling_time_range = (1e9, 1e9)
    cfg.commands.base_velocity.heading_command = False
    cfg.commands.base_velocity.ranges.lin_vel_x = (args.velocity[0], args.velocity[0])
    cfg.commands.base_velocity.ranges.lin_vel_y = (args.velocity[1], args.velocity[1])
    cfg.commands.base_velocity.ranges.ang_vel_z = (args.velocity[2], args.velocity[2])
    cfg.commands.base_velocity.low_vel_env_lin_x_ranges = (args.velocity[0], args.velocity[0])
    # Fixed commands must not be zeroed by the command sampler's standing-env mask.
    cfg.commands.base_velocity.rel_standing_envs = 0.
    cfg.commands.base_velocity.min_command_norm = 0.
    env = HimlocoVecEnvWrapper(gym.make('Unitree-Go1-Velocity-PD-Play', cfg=cfg), history_length=5)
    try:
        base = env.unwrapped
        robot = base.scene['robot']
        lab_names = list(robot.joint_names)
        if lab_names != list(GO1_JOINT_NAMES):
            raise AssertionError('Live Lab joint order differs from Go1 policy order')
        sdk_indices = [lab_names.index(name) for name in SDK_JOINT_NAMES]
        term = base.action_manager.get_term('JointPositionAction')
        if term.action_dim != 12 or not cfg.actions.JointPositionAction.use_default_offset:
            raise AssertionError('Unexpected native position action interface')
        q0 = array(robot.data.default_joint_pos.torch)
        np.testing.assert_allclose(q0, np.broadcast_to(q0[0], q0.shape), rtol=0, atol=0)
        command_term = base.command_manager.get_term('base_velocity')
        command = torch.tensor(args.velocity, dtype=torch.float32, device=env.device).repeat(args.num_envs, 1)
        command_term.vel_command_b.copy_(command)
        # Seed exactly one current frame and five zero past frames, as wrapper.reset does.
        observations = base.observation_manager.compute()
        env.obs_history_buf = torch.cat((observations['policy'],
            torch.zeros(args.num_envs, 225, device=env.device)), dim=-1)
        env.privileged_obs_history_buf = observations['critic'].clone()
        records = []
        episode = np.zeros(args.num_envs, dtype=np.int64)
        history_reset = np.ones(args.num_envs, dtype=bool)
        max_target_error = 0.
        resets = 0
        for step in range(args.steps):
            history = array(env.get_observations())
            quaternion_xyzw = array(robot.data.root_quat_w.torch)
            record = dict(
                step=np.full(args.num_envs, step, dtype=np.int64),
                env_id=np.arange(args.num_envs, dtype=np.int64),
                episode_id=episode.copy(), history_reset=history_reset.copy(),
                q_sdk=array(robot.data.joint_pos.torch)[:, sdk_indices],
                dq_sdk=array(robot.data.joint_vel.torch)[:, sdk_indices],
                quaternion_wxyz=quaternion_xyzw[:, [3, 0, 1, 2]],
                gyro_sensor=array(robot.data.root_ang_vel_b.torch),
                commands=array(base.command_manager.get_command('base_velocity')),
                previous_action_lab=array(base.action_manager.action),
                expected_frame=history[:, :45].copy(), expected_history=history,
            )
            with torch.no_grad():
                action = model(torch.from_numpy(history))
            if tuple(action.shape) != (args.num_envs, 12) or not torch.isfinite(action).all():
                raise AssertionError('Expected finite batched 270D -> 12D JIT inference')
            record['raw_action_lab'] = array(action)
            with torch.no_grad():
                _, _, _, done, _, reset_ids, _ = env.step(action.to(env.device))
            # JointAction.reset clears raw_actions only; processed_actions retains
            # this physics step's affine+clipped target even for auto-reset envs.
            # Read the real term buffer, rather than reconstructing our expected value.
            target = array(term.processed_actions)
            affine_target = np.clip(q0 + .25 * record['raw_action_lab'], -100., 100.)
            np.testing.assert_allclose(target, affine_target, rtol=0, atol=2e-6)
            max_target_error = max(max_target_error, float(np.max(np.abs(target-affine_target))))
            record['expected_target_lab'] = target
            record['done_after_action'] = array(done).astype(bool)
            for name, value in record.items():
                if value.shape[0] != args.num_envs or not np.isfinite(value).all():
                    raise AssertionError(f'Invalid synchronized field {name}: {value.shape}')
            if history.shape != (args.num_envs, 270):
                raise AssertionError('Expected native current-first six-frame history')
            records.append(record)
            ids = array(reset_ids).astype(np.int64)
            history_reset = np.zeros(args.num_envs, dtype=bool)
            history_reset[ids] = True
            episode[ids] += 1
            resets += len(ids)
        # Flatten time-major then env; replay must keep separate per-env histories.
        data = {key: np.concatenate([record[key] for record in records], axis=0) for key in records[0]}
        data.update(lab_joint_names=np.asarray(lab_names), sdk_joint_names=np.asarray(SDK_JOINT_NAMES),
                    default_joint_pos_lab=q0[0], default_joint_vel_lab=array(robot.data.default_joint_vel.torch)[0])
        np.savez_compressed(output/'reference.npz', **data)
        source_paths = [Path(inspect.getsourcefile(JointAction)),
            Path(inspect.getsourcefile(type(base.observation_manager))),
            Path(inspect.getsourcefile(type(env))), Path(inspect.getsourcefile(Go1PDPlayEnvCfg))]
        repo = Path(__file__).resolve().parents[2]
        metadata = dict(
            status='passed', task='Unitree-Go1-Velocity-PD-Play', seed=args.seed,
            steps=args.steps, num_envs=args.num_envs, samples=args.steps*args.num_envs,
            policy=str(policy_path), policy_sha256=policy_hash,
            checkpoint_sha256=manifest.get('checkpoint_sha256'),
            native_lab_source_sha=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
            python=platform.python_version(), torch=torch.__version__, isaacsim=version('isaacsim'),
            isaaclab_distribution=version('isaaclab'), policy_dt=base.step_dt,
            lab_joint_names=lab_names, sdk_joint_names=list(SDK_JOINT_NAMES),
            default_joint_pos_lab=q0[0].tolist(), command=list(args.velocity),
            history_order='current,previous_1,...,previous_5; reset past frames zero',
            previous_action='native action_manager.action, before observation clip +/-100',
            target_processing='clip(default_joint_pos + 0.25 * raw_action, -100, 100)',
            synchronized_state='pre-action; expected target belongs to the same raw action',
            source_files={str(path): sha256(path) for path in source_paths},
            target_formula_max_abs_error=max_target_error, resets=resets,
            reference_sha256=sha256(output/'reference.npz'),
            motor_commands_published=False,
        )
        (output/'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
        print('LAB_DEPLOY_REFERENCE_OK', json.dumps(metadata), flush=True)
    finally:
        env.close()


try:
    main()
except BaseException:
    import traceback
    traceback.print_exc()
    launcher.app.close(exit_code=1)
else:
    launcher.app.close()
