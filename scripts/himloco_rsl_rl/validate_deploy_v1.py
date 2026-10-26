"""Validate profile isolation and actual PhysX non-foot contact rewards."""
import argparse
from pathlib import Path
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--num_envs', type=int, default=32)
parser.add_argument('--output', default='artifacts/go1_deploy_v1/validation.json')
AppLauncher.add_app_launcher_args(parser)
parser.add_argument('--headless', action='store_true')
args = parser.parse_args()
launcher = AppLauncher(args)

import json
import gymnasium as gym
import torch
import himloco_lab.tasks  # noqa: F401
from himloco_lab.rsl_rl import HimlocoVecEnvWrapper
from himloco_lab.tasks.locomotion import mdp
from himloco_lab.tasks.locomotion.robots.go1.pd_env_cfg import Go1PDEnvCfg
from himloco_lab.tasks.locomotion.robots.go1.velocity_env_cfg import Go1EnvCfg
from himloco_lab.tasks.locomotion.robots.go1.deploy_v1_env_cfg import Go1PDDeployV1EnvCfg, Go1PDDeployV1PlayEnvCfg
from himloco_lab.tasks.locomotion.agents.himloco_rsl_rl_cfg import Go1PDPPORunnerCfg, Go1PDDeployV1PPORunnerCfg


def differences(a, b, path=''):
    if isinstance(a, dict) and isinstance(b, dict):
        result = []
        for key in sorted(set(a) | set(b)):
            label = (path + '.' if path else '') + key
            result += [label] if key not in a or key not in b else differences(a[key], b[key], label)
        return result
    return [] if repr(a) == repr(b) else [path]


def main():
    output = Path(args.output)
    if output.exists():
        raise FileExistsError('Choose a new report path; previous results are preserved')
    before_pd = Go1PDEnvCfg().to_dict()
    before_net = Go1EnvCfg().to_dict()
    cfg, play = Go1PDDeployV1EnvCfg(), Go1PDDeployV1PlayEnvCfg()
    delta = differences(before_pd, cfg.to_dict())
    assert set(delta) == {
        'rewards.base_height_l2.params.target_height', 'rewards.base_height_l2.weight',
        'rewards.undesired_leg_contacts', 'commands.base_velocity.min_command_norm',
        'commands.base_velocity.rel_heading_envs'}, delta
    assert before_pd == Go1PDEnvCfg().to_dict()
    assert before_net == Go1EnvCfg().to_dict()
    assert cfg.commands.base_velocity.heading_command and cfg.commands.base_velocity.rel_heading_envs == .5
    assert not play.commands.base_velocity.heading_command and play.commands.base_velocity.min_command_norm == 0.
    assert differences(Go1PDPPORunnerCfg().to_dict(), Go1PDDeployV1PPORunnerCfg().to_dict()) == ['experiment_name']
    for name in ('Unitree-Go1-Velocity-PD-DeployV1', 'Unitree-Go1-Velocity-PD-DeployV1-Play'):
        assert name in gym.registry
    # Diagnostics only; the separate training smoke uses the unchanged rough terrain.
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    cfg.seed = 1
    cfg.scene.terrain.terrain_type = 'plane'
    cfg.scene.terrain.terrain_generator = None
    cfg.scene.terrain.max_init_terrain_level = None
    cfg.curriculum.terrain_levels = None
    cfg.curriculum.lin_vel_cmd_levels = None
    cfg.observations.policy.enable_corruption = False
    for name in ('physics_material', 'add_base_mass', 'randomize_rigid_body_com', 'external_force', 'push_robot'):
        setattr(cfg.events, name, None)
    cfg.events.reset_robot_joints.params.update(position_range=(1., 1.), velocity_range=(0., 0.))
    cfg.events.reset_base.params['pose_range'] = {k:(0.,0.) for k in ('x','y','yaw')}
    cfg.events.reset_base.params['velocity_range'] = {k:(0.,0.) for k in ('x','y','z','roll','pitch','yaw')}
    env = HimlocoVecEnvWrapper(gym.make('Unitree-Go1-Velocity-PD-DeployV1', cfg=cfg), history_length=5)
    try:
        base, robot = env.unwrapped, env.unwrapped.scene['robot']
        assert (env.num_one_step_obs, env.num_obs, env.num_privileged_obs, env.num_actions) == (45,270,238,12)
        actuator = robot.actuators['base_legs']
        assert type(actuator).__name__ == 'IdealPDActuator'
        torch.testing.assert_close(actuator.stiffness, torch.full_like(actuator.stiffness,40.))
        torch.testing.assert_close(actuator.damping, torch.ones_like(actuator.damping))
        term = base.reward_manager.get_term_cfg('undesired_leg_contacts')
        sensor_cfg = term.params['sensor_cfg']
        sensor = base.scene.sensors[sensor_cfg.name]
        selected = [sensor.body_names[i] for i in sensor_cfg.body_ids]
        expected = {f'{leg}_{joint}' for leg in ('FL','FR','RL','RR') for joint in ('thigh','calf')}
        assert set(selected) == expected and len(selected) == 8, selected
        assert all('_foot' not in name for name in selected)
        heading = base.command_manager.get_term('base_velocity').is_heading_env
        assert heading.any() and (~heading).any(), 'Expected both heading and direct-yaw training groups'
        standing_max = 0.
        with torch.no_grad():
            for _ in range(100):
                obs, privileged, reward, _, _, _, _ = env.step(torch.zeros(env.num_envs,12,device=env.device))
                assert torch.isfinite(obs).all() and torch.isfinite(privileged).all() and torch.isfinite(reward).all()
            counts = mdp.undesired_contacts(base, threshold=1., sensor_cfg=sensor_cfg)
            standing_max = float(counts.max())
            assert standing_max == 0., 'Normal foot stance must not incur a leg-contact penalty'
            # Deliberately lower the robot into the floor without manager auto-reset,
            # then inspect actual PhysX calf/thigh forces. No sensor buffers are forged.
            pose = robot.data.root_pose_w.torch.clone()
            pose[:,2] = .10
            robot.write_root_pose_to_sim(pose)
            robot.write_root_velocity_to_sim(torch.zeros(env.num_envs,6,device=env.device))
            for _ in range(4):
                base.scene.write_data_to_sim()
                base.sim.step(render=False)
                base.scene.update(base.physics_dt)
            forces = sensor.data.net_normal_forces_w_history.torch[:,:,sensor_cfg.body_ids]
            expected_counts = (forces.norm(dim=-1).amax(dim=1)>1.).sum(dim=1)
            actual_counts = mdp.undesired_contacts(base, threshold=1., sensor_cfg=sensor_cfg)
            torch.testing.assert_close(actual_counts, expected_counts)
            assert actual_counts.max() > 0, 'Actual non-foot collision must activate the new reward'
        record = dict(status='passed', parameter_changes=delta, baseline_configs_unchanged=True,
            runner_changes=['experiment_name'], num_envs=env.num_envs, observation_dims=[45,270,238], actions=12,
            selected_contact_bodies=selected, standing_contact_count_max=standing_max,
            lowered_robot_contact_count_max=int(actual_counts.max()), force_source='actual PhysX normal force history',
            heading_env_count=int(heading.sum()), direct_yaw_env_count=int((~heading).sum()),
            physics_dt=base.physics_dt, policy_dt=base.step_dt, actuator=type(actuator).__name__, kp=40,kd=1,
            height_error_5cm_weighted_penalty_before_dt=-2*(.05**2),
            diagnostic_overrides='plane/nominal reset/no corruption or randomization; not training defaults',
            motor_commands_published=False)
        output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(json.dumps(record,indent=2)+'\n')
        print('DEPLOY_V1_VALIDATION_OK',json.dumps(record),flush=True)
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
