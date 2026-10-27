"""Paired flat-ground evaluation of two Go1 HIM JITs, without training.

Both policies use one deterministic evaluation scene and explicit physical
commands (including low speeds). Report first-episode states only: auto-reset
states never become terminal gait samples. This is simulation, not hardware.
"""
import argparse
import hashlib
import json
from pathlib import Path
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--baseline', required=True)
parser.add_argument('--candidate', required=True, action='append',
                    help='Candidate JIT; repeat to compare saved checkpoints without retraining')
parser.add_argument('--duration', type=float, default=20.)
parser.add_argument('--replicas', type=int, default=4)
parser.add_argument('--output', default='artifacts/go1_deploy_v1/evaluation_20261027/native_lab')
AppLauncher.add_app_launcher_args(parser)
parser.add_argument('--headless', action='store_true')
args = parser.parse_args()
launcher = AppLauncher(args)

import gymnasium as gym
import numpy as np
import torch
import himloco_lab.tasks  # noqa: F401
from himloco_lab.rsl_rl import HimlocoVecEnvWrapper
from himloco_lab.tasks.locomotion.robots.go1.deploy_v1_env_cfg import Go1PDDeployV1PlayEnvCfg

COMMANDS = [[0.,0.,0.],[.3,0.,0.],[-.3,0.,0.],[.6,0.,0.],[-.6,0.,0.],
            [0.,.3,0.],[0.,0.,.3],[.1,0.,0.],[-.1,0.,0.]]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@torch.inference_mode()
def main():
    if args.duration <= 2 or args.replicas < 1:
        raise ValueError('Use duration > 2s and replicas >= 1')
    output = Path(args.output)
    if (output/'comparison.json').exists():
        raise FileExistsError('Choose a new output path; prior comparisons are preserved')
    torch.set_num_threads(1)
    cfg = Go1PDDeployV1PlayEnvCfg()
    cfg.seed = 1
    cfg.scene.num_envs = len(COMMANDS)*args.replicas
    cfg.sim.device = args.device
    cfg.scene.terrain.terrain_type = 'plane'
    cfg.scene.terrain.terrain_generator = None
    cfg.scene.terrain.max_init_terrain_level = None
    cfg.episode_length_s = args.duration+1.
    cfg.curriculum.terrain_levels = None
    cfg.curriculum.lin_vel_cmd_levels = None
    for name in ('physics_material','add_base_mass','randomize_rigid_body_com','external_force','push_robot'):
        setattr(cfg.events,name,None)
    cfg.events.reset_robot_joints.params.update(position_range=(1.,1.),velocity_range=(0.,0.))
    cfg.events.reset_base.params['pose_range'] = {k:(0.,0.) for k in ('x','y','yaw')}
    cfg.events.reset_base.params['velocity_range'] = {k:(0.,0.) for k in ('x','y','z','roll','pitch','yaw')}
    cfg.commands.base_velocity.resampling_time_range = (1e9,1e9)
    cfg.commands.base_velocity.min_command_norm = 0.
    env = HimlocoVecEnvWrapper(gym.make('Unitree-Go1-Velocity-PD-DeployV1-Play',cfg=cfg),history_length=5)
    try:
        base, robot = env.unwrapped, env.unwrapped.scene['robot']
        sensor = base.scene.sensors['contact_forces']
        names = [f'{leg}_{part}' for leg in ('FL','FR','RL','RR') for part in ('thigh','calf')]
        contact_ids = [sensor.body_names.index(n) for n in names]
        knee_ids = [robot.body_names.index(f'{leg}_calf') for leg in ('FL','FR','RL','RR')]
        trunk = sensor.body_names.index('trunk')
        commands = torch.tensor(COMMANDS,device=env.device).repeat_interleave(args.replicas,dim=0)
        results = []
        state_hash = None
        policies = [('baseline', args.baseline)] + [
            ('candidate' if len(args.candidate) == 1 else f'candidate_{i}', path)
            for i, path in enumerate(args.candidate)]
        for label,path in policies:
            model = torch.jit.load(path,map_location=env.device).eval()
            base.seed(1)
            env.reset()
            initial = torch.cat((robot.data.root_pose_w.torch,robot.data.root_vel_w.torch,
                                 robot.data.joint_pos.torch,robot.data.joint_vel.torch),dim=-1).cpu().numpy()
            signature = hashlib.sha256(initial.tobytes()).hexdigest()
            if state_hash is not None and signature != state_hash:
                raise AssertionError('Nominal initial states are not identical')
            state_hash = signature
            base.command_manager.get_term('base_velocity').vel_command_b.copy_(commands)
            obs = base.observation_manager.compute()
            env.obs_history_buf = torch.cat((obs['policy'],torch.zeros(env.num_envs,225,device=env.device)),dim=-1)
            env.privileged_obs_history_buf = obs['critic'].clone()
            active = torch.ones(env.num_envs,dtype=torch.bool,device=env.device)
            survival = torch.full((env.num_envs,),args.duration,device=env.device)
            bad_time = torch.zeros(env.num_envs,device=env.device)
            samples = []
            with torch.inference_mode():
                for step in range(round(args.duration/base.step_dt)):
                    action = model(env.get_observations())
                    if not torch.isfinite(action).all():
                        raise FloatingPointError('Non-finite action')
                    _,_,reward,done,_,_,_ = env.step(action)
                    if not torch.isfinite(reward).all():
                        raise FloatingPointError('Non-finite reward')
                    height = robot.data.root_pos_w.torch[:,2]
                    gravity = robot.data.projected_gravity_b.torch
                    force = sensor.data.net_normal_forces_w_history.torch.norm(dim=-1).amax(dim=1)
                    bad = (height<.16) | (-gravity[:,2]<np.cos(1.)) | (force[:,trunk]>1.)
                    bad_time = torch.where(bad,bad_time+base.step_dt,0.)
                    ended = active & (done.bool() | (bad_time>=.1))
                    survival[ended] = (step+1)*base.step_dt
                    # Done environments have already reset: exclude them before reading gait states.
                    valid = active & ~done.bool()
                    samples.append(dict(valid=valid.clone(),height=height.clone(),
                        root_position=robot.data.root_pos_w.torch.clone(),
                        knee_z=robot.data.body_pos_w.torch[:,knee_ids,2].clone(),
                        velocity=robot.data.root_lin_vel_b.torch.clone(),
                        omega=robot.data.root_ang_vel_b.torch.clone(),
                        quaternion=robot.data.root_quat_w.torch.clone(),
                        leg_contact=(force[:,contact_ids]>1.).clone()))
                    active &= ~ended
            trace = {k:torch.stack([r[k] for r in samples]).cpu().numpy() for k in samples[0]}
            trace['survival_s'] = survival.cpu().numpy()
            trace['survived_horizon'] = active.cpu().numpy()
            trace['commands'] = commands.cpu().numpy()
            output.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(output/f'{label}.npz',**trace)
            elapsed = (np.arange(len(samples))+1)*base.step_dt
            cases = []
            for index,command in enumerate(COMMANDS):
                group = slice(index*args.replicas,(index+1)*args.replicas)
                valid = trace['valid'][:,group] & (elapsed>2)[:,None]
                velocity = trace['velocity'][:,group][valid]
                omega = trace['omega'][:,group][valid]
                height = trace['height'][:,group][valid]
                knee = trace['knee_z'][:,group][valid]
                contact = trace['leg_contact'][:,group][valid]
                quat = trace['quaternion'][:,group][valid] # Lab3 XYZW
                x,y,z,w = quat.T
                roll = np.arctan2(2*(w*x+y*z),1-2*(x*x+y*y))
                pitch = np.arcsin(np.clip(2*(w*y-z*x),-1,1))
                heading_changes, cross_tracks = [], []
                for replica in range(args.replicas):
                    indices = np.flatnonzero(valid[:, replica])
                    if len(indices) < 2:
                        continue
                    q = trace['quaternion'][indices, index*args.replicas+replica]
                    qx,qy,qz,qw = q.T
                    yaw = np.unwrap(np.arctan2(2*(qw*qz+qx*qy),1-2*(qy*qy+qz*qz)))
                    p = trace['root_position'][indices, index*args.replicas+replica]
                    delta = p[-1] - p[0]
                    heading_changes.append(float(np.rad2deg(yaw[-1]-yaw[0])))
                    cross_tracks.append(float(-np.sin(yaw[0])*delta[0]+np.cos(yaw[0])*delta[1]))
                cases.append(dict(command=command,survivors=int(trace['survived_horizon'][group].sum()),
                    replicas=args.replicas,valid_samples=int(valid.sum()),
                    mean_velocity=velocity.mean(0).tolist() if len(velocity) else None,
                    mean_wz=float(omega[:,2].mean()) if len(omega) else None,
                    linear_rmse=float(np.sqrt(np.mean(np.sum((velocity[:,:2]-command[:2])**2,axis=1)))) if len(velocity) else None,
                    yaw_rmse=float(np.sqrt(np.mean((omega[:,2]-command[2])**2))) if len(omega) else None,
                    base_height_mean=float(height.mean()) if len(height) else None,
                    base_height_p5=float(np.percentile(height,5)) if len(height) else None,
                    rear_knee_origin_z_p5=float(np.percentile(knee[:,2:],5)) if len(knee) else None,
                    rear_knee_origin_z_min=float(knee[:,2:].min()) if len(knee) else None,
                    leg_contact_fraction=float(contact.any(axis=1).mean()) if len(contact) else None,
                    rear_leg_contact_fraction=float(contact[:,4:].any(axis=1).mean()) if len(contact) else None,
                    heading_change_deg_mean=float(np.mean(heading_changes)) if heading_changes else None,
                    heading_change_abs_deg_mean=float(np.mean(np.abs(heading_changes))) if heading_changes else None,
                    cross_track_displacement_mean_m=float(np.mean(cross_tracks)) if cross_tracks else None,
                    cross_track_displacement_abs_mean_m=float(np.mean(np.abs(cross_tracks))) if cross_tracks else None,
                    abs_roll_pitch_p95=np.percentile(np.abs(np.column_stack((roll,pitch))),95,axis=0).tolist() if len(roll) else None))
            results.append(dict(label=label,policy=str(Path(path).resolve()),policy_sha256=sha256(path),
                initial_state_sha256=signature,cases=cases))
            print('PAIRED_POLICY_COMPLETE',label,flush=True)
        report=dict(status='passed',simulator='Isaac Lab3 PhysX',duration_s=args.duration,seed=1,
            warmup_excluded_s=2.,paired_initial_states=True,results=results,
            limitations='Flat/no randomization. Replicas are translated nominal copies, not a robustness confidence interval. '
            'Knee metric is joint-origin height, not collision-surface clearance. Contacts use >1N normal force '
            'history at 50Hz samples. No hardware conclusion or per-reward causality claim.',
            drift_definition='Signed/absolute heading change and actual root displacement across the starting '
            'heading, from the first valid sample after 2s to the final valid first-episode sample. '
            'Heading change is only unwanted drift when the requested yaw rate is zero.',
            motor_commands_published=False)
        (output/'comparison.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        print('PAIRED_EVALUATION_OK',output/'comparison.json',flush=True)
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
