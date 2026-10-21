"""Paired direct/adapted legacy evaluation using physical state, never reward totals."""
import argparse
import json
from pathlib import Path
import sys
import time
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', required=True)
parser.add_argument('--mode', choices=['direct', 'adapted', 'both'], default='both')
parser.add_argument('--terrain', choices=['plane', 'rough'], default='plane')
parser.add_argument('--num_envs', type=int, default=40)
parser.add_argument('--duration', type=float, default=20.)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--velocity', type=float, nargs=3, default=None)
parser.add_argument('--output', default='artifacts/round2/legacy_plane')
parser.add_argument('--real_time', action='store_true')
AppLauncher.add_app_launcher_args(parser)
parser.add_argument('--headless', action='store_true')
args = parser.parse_args()
if not args.headless and args.visualizer is None:
    args.visualizer = ['kit']; args.visualizer_explicit = True
launcher = AppLauncher(args)
sys.argv = [sys.argv[0]]

import gymnasium as gym
import torch
import himloco_lab.tasks
from himloco_lab.envs import HimlocoManagerBasedRLEnv
from himloco_lab.rsl_rl import HimlocoVecEnvWrapper
from himloco_lab.tasks.locomotion.robots.go1.velocity_env_cfg import Go1PlayEnvCfg
from himloco_lab.inference.legacy_go1 import LegacyGo1ObservationAdapter, LEGACY_JOINT_NAMES
from himloco_lab.utils.checkpoint import load_inference_model
from isaaclab.utils.math import euler_xyz_from_quat


class EvaluationEnv(HimlocoManagerBasedRLEnv):
    """Capture true pre-reset states so failures do not contaminate physical metrics."""
    def state(self):
        robot = self.scene['robot']
        return {name: getattr(robot.data, name).torch.clone() for name in
                ('root_pos_w', 'root_lin_vel_b', 'root_ang_vel_b', 'root_quat_w')}

    def _reset_idx(self, env_ids):
        if getattr(self, '_in_him_step', False):
            self.terminal_state = (env_ids.clone(), {k: v[env_ids].clone() for k, v in self.state().items()})
        super()._reset_idx(env_ids)


def main():
    torch.set_num_threads(4)
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    model, audit = load_inference_model(args.checkpoint, args.device)
    cfg = Go1PlayEnvCfg(); cfg.seed = args.seed; cfg.scene.num_envs = args.num_envs
    cfg.episode_length_s = args.duration + 1
    cfg.sim.device = args.device
    cfg.curriculum.terrain_levels = None; cfg.curriculum.lin_vel_cmd_levels = None
    for name in ('physics_material', 'add_base_mass', 'randomize_rigid_body_com', 'external_force', 'push_robot'):
        setattr(cfg.events, name, None)
    cfg.events.reset_robot_joints.params['position_range'] = (1., 1.)
    cfg.events.reset_base.params['pose_range'] = {'x': (0., 0.), 'y': (0., 0.), 'yaw': (0., 0.)}
    cfg.commands.base_velocity.resampling_time_range = (1e9, 1e9)
    cfg.commands.base_velocity.heading_command = False
    cfg.scene.terrain.terrain_generator.seed = args.seed
    if args.terrain == 'plane':
        cfg.scene.terrain.terrain_type = 'plane'; cfg.scene.terrain.terrain_generator = None
    else:
        cfg.scene.terrain.terrain_generator.num_rows = 3
        cfg.scene.terrain.terrain_generator.num_cols = 10
        cfg.scene.terrain.terrain_generator.difficulty_range = (0., .5)
        cfg.scene.terrain.max_init_terrain_level = 1
    if not args.headless:
        from isaaclab_visualizers.kit import KitVisualizerCfg
        cfg.sim.visualizer_cfgs = [KitVisualizerCfg(origin_type='asset', origin_track_path='robot',
                                eye=(3., -3., 2.), lookat=(0., 0., 0.))]
    base = EvaluationEnv(cfg)
    env = HimlocoVecEnvWrapper(base, history_length=5)
    robot = base.scene['robot']
    action_names = robot.joint_names  # explicitly ordered by Go1EnvCfg for both actions and joint observations
    assert list(base.cfg.actions.JointPositionAction.joint_names) == action_names
    defaults = dict(zip(robot.joint_names, robot.data.default_joint_pos.torch[0].cpu().tolist()))
    assert base.cfg.actions.JointPositionAction.use_default_offset
    adapter = LegacyGo1ObservationAdapter(robot.joint_names, action_names, defaults,
        lab_action_scale=base.cfg.actions.JointPositionAction.scale, device=env.device)
    command_list = [args.velocity] if args.velocity is not None else [[0,0,0],[.5,0,0],[.7,0,0],[0,0,.3],[.7,0,.3]]
    command_ids = torch.arange(args.num_envs, device=env.device) % len(command_list)
    commands = torch.tensor(command_list, device=env.device, dtype=torch.float32)[command_ids]
    modes = ['direct', 'adapted'] if args.mode == 'both' else [args.mode]
    results = []
    initial_reference = None
    for mode in modes:
        # Pair the exact same native reset state. No noise, randomization or stochastic policy.
        base.seed(args.seed); env.reset()
        base.command_manager.get_term('base_velocity').vel_command_b.copy_(commands)
        observations = base.observation_manager.compute()
        env.obs_history_buf = torch.cat((observations['policy'], torch.zeros(args.num_envs, 225, device=env.device)), -1)
        env.privileged_obs_history_buf = observations['critic'].clone()
        initial = base.state()
        if initial_reference is not None:
            for key in initial:
                torch.testing.assert_close(initial[key], initial_reference[key], rtol=0, atol=0)
        initial_reference = initial
        active = torch.ones(args.num_envs, dtype=torch.bool, device=env.device)
        failures = torch.zeros_like(active); truncations = torch.zeros_like(active)
        elapsed = torch.zeros(args.num_envs, device=env.device)
        last_pos = initial['root_pos_w'].clone(); path = torch.zeros_like(elapsed)
        forward_distance = torch.zeros_like(elapsed)
        velocity_sum = torch.zeros(args.num_envs, 3, device=env.device)
        angular_sum = torch.zeros(args.num_envs, device=env.device)
        lin_error_sq = torch.zeros_like(elapsed); yaw_error_sq = torch.zeros_like(elapsed)
        roll_min = torch.full_like(elapsed, float('inf')); roll_max = -roll_min
        pitch_min = roll_min.clone(); pitch_max = roll_max.clone()
        action_peak = torch.zeros_like(elapsed); clipped_steps = torch.zeros_like(elapsed)
        samples_lab = []; samples_policy = []
        reset_ids = torch.arange(args.num_envs, device=env.device)
        steps = round(args.duration / base.step_dt)
        with torch.no_grad():
            for step in range(steps):
                started = time.monotonic()
                lab_obs = env.get_observations()
                policy_obs = adapter.adapt_history(lab_obs, reset_ids) if mode == 'adapted' else lab_obs
                old_action = model.act_inference(policy_obs)
                lab_action = adapter.to_lab_action(old_action) if mode == 'adapted' else old_action.clamp(-100, 100)
                if step % 10 == 0 and len(samples_lab) < 128:
                    samples_lab.append(lab_obs[::4].clone()); samples_policy.append(policy_obs[::4].clone())
                base.terminal_state = None
                _, _, _, done, _, reset_ids, _ = env.step(lab_action)
                state = base.state()
                if base.terminal_state is not None:
                    ids, terminal = base.terminal_state
                    for key in state:
                        state[key][ids] = terminal[key]
                if not all(torch.isfinite(x).all() for x in state.values()):
                    raise ValueError('Non-finite physical state')
                delta = state['root_pos_w'] - last_pos
                path += delta[:, :2].norm(dim=-1) * active
                last_pos[active] = state['root_pos_w'][active]
                elapsed += active * base.step_dt
                velocity_sum += state['root_lin_vel_b'] * active[:, None]
                angular_sum += state['root_ang_vel_b'][:, 2] * active
                forward_distance += state['root_lin_vel_b'][:, 0] * active * base.step_dt
                lin_error_sq += (state['root_lin_vel_b'][:, :2]-commands[:, :2]).square().sum(-1) * active
                yaw_error_sq += (state['root_ang_vel_b'][:, 2]-commands[:, 2]).square() * active
                roll, pitch, _ = euler_xyz_from_quat(state['root_quat_w'])
                roll = (roll + torch.pi) % (2*torch.pi) - torch.pi
                pitch = (pitch + torch.pi) % (2*torch.pi) - torch.pi
                roll_min = torch.where(active, torch.minimum(roll_min, roll), roll_min)
                roll_max = torch.where(active, torch.maximum(roll_max, roll), roll_max)
                pitch_min = torch.where(active, torch.minimum(pitch_min, pitch), pitch_min)
                pitch_max = torch.where(active, torch.maximum(pitch_max, pitch), pitch_max)
                action_peak = torch.where(active, torch.maximum(action_peak, old_action.abs().amax(-1)), action_peak)
                clipped_steps += ((old_action.abs() > 100).any(-1) & active)
                failures |= active & base.reset_terminated
                truncations |= active & base.reset_time_outs & ~base.reset_terminated
                active &= ~done
                if args.real_time:
                    time.sleep(max(0., base.step_dt-(time.monotonic()-started)))
        counts = (elapsed/base.step_dt).clamp_min(1)
        for index, command in enumerate(command_list):
            mask = command_ids == index
            result = dict(mode=mode, terrain=args.terrain, command=command, episodes=int(mask.sum()),
                          survival_s_mean=float(elapsed[mask].mean()),
                          survival_s_min=float(elapsed[mask].min()),
                          fall_rate=float(failures[mask].float().mean()),
                          termination_rate=float((failures|truncations)[mask].float().mean()),
                          completed_horizon_rate=float(active[mask].float().mean()),
                          world_x_displacement_m_mean=float((last_pos-initial['root_pos_w'])[mask, 0].mean()),
                          body_forward_distance_m_mean=float(forward_distance[mask].mean()),
                          horizontal_path_m_mean=float(path[mask].mean()),
                          measured_base_velocity_mean=(velocity_sum/counts[:,None])[mask].mean(0).tolist(),
                          measured_yaw_rate_mean=float((angular_sum/counts)[mask].mean()),
                          linear_tracking_rmse=float(torch.sqrt(lin_error_sq[mask].sum()/counts[mask].sum())),
                          yaw_tracking_rmse=float(torch.sqrt(yaw_error_sq[mask].sum()/counts[mask].sum())),
                          roll_range_rad=[float(roll_min[mask].min()),float(roll_max[mask].max())],
                          pitch_range_rad=[float(pitch_min[mask].min()),float(pitch_max[mask].max())],
                          max_abs_action=float(action_peak[mask].max()), clipped_action_steps=int(clipped_steps[mask].sum()))
            results.append(result); print('EVAL_CASE', json.dumps(result), flush=True)
        torch.save(dict(lab_observations=torch.cat(samples_lab).cpu(),
                        policy_observations=torch.cat(samples_policy).cpu()), output / f'{mode}_observations.pt')
    report = dict(checkpoint=audit, seed=args.seed, duration=args.duration, num_envs=args.num_envs,
                  gui=not args.headless, paired_initial_states=True, legacy_joint_names=LEGACY_JOINT_NAMES,
                  lab_joint_names=robot.joint_names, lab_default_positions=defaults,
                  evaluation='One first episode per environment, censored at horizon; pre-reset physical states.',
                  results=results)
    (output/'results.json').write_text(json.dumps(report, indent=2))
    print('LEGACY_EVAL_OK', str(output/'results.json'), flush=True)
    env.close()

try:
    main()
except BaseException:
    import traceback
    traceback.print_exc(); launcher.app.close(exit_code=1)
else:
    launcher.app.close()
