"""Reproducible Go1 integration audit including real PPO, autoreset and exports."""
import argparse
import sys
from pathlib import Path
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument('--num_envs', type=int, default=32)
parser.add_argument('--output', default='logs/go1_validation')
AppLauncher.add_app_launcher_args(parser)
parser.add_argument('--headless', action='store_true')
args = parser.parse_args()
launcher = AppLauncher(args)

import json
import unittest
import gymnasium as gym
import numpy as np
import torch
import onnxruntime as ort
import himloco_lab.tasks
from himloco_lab.rsl_rl import HIMOnPolicyRunner, HimlocoVecEnvWrapper
from himloco_lab.tasks.locomotion.robots.go1.velocity_env_cfg import Go1EnvCfg, GO1_JOINT_NAMES
from himloco_lab.tasks.locomotion.agents.himloco_rsl_rl_cfg import Go1PPORunnerCfg
from himloco_lab.utils import export_himloco_policy_as_jit, export_himloco_policy_as_onnx
from isaaclab.utils.math import quat_apply_inverse


def main():
    torch.set_num_threads(4)
    suite = unittest.defaultTestLoader.discover(str(Path(__file__).resolve().parents[2] / 'tests'), pattern='test_him_contracts.py')
    if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():
        raise AssertionError('HIM contract tests failed')
    cfg = Go1EnvCfg(); cfg.scene.num_envs = args.num_envs; cfg.seed = 1
    cfg.scene.terrain.terrain_generator.seed = cfg.seed
    cfg.observations.policy.enable_corruption = False
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    env = HimlocoVecEnvWrapper(gym.make('Unitree-Go1-Velocity', cfg=cfg), history_length=5)
    robot = env.unwrapped.scene['robot']
    assert (env.num_one_step_obs, env.num_obs, env.num_privileged_obs, env.num_actions) == (45, 270, 238, 12)
    assert robot.joint_names == GO1_JOINT_NAMES, robot.joint_names
    expected_default = torch.tensor([.1, -.1, .1, -.1, .8, .8, 1., 1., -1.5, -1.5, -1.5, -1.5], device=env.device)
    torch.testing.assert_close(robot.data.default_joint_pos.torch[0], expected_default)
    assert 'trunk' in robot.body_names
    scanner = env.unwrapped.scene['height_scanner']
    assert scanner.data.ray_hits_w.torch.shape == (args.num_envs, 187, 3)
    assert torch.isfinite(scanner.data.ray_hits_w.torch).all()
    torch.testing.assert_close(scanner.data.pos_w.torch[:, 2], robot.data.root_pos_w.torch[:, 2])
    assert (env.get_privileged_observations()[:, 51:] < 4.9).any()
    from himloco_lab.tasks.locomotion.robots.go2.velocity_env_cfg import RobotEnvCfg as Go2EnvCfg
    go2 = Go2EnvCfg()
    assert 'GO2HV' in go2.scene.robot.actuators
    assert go2.scene.height_scanner.prim_path.endswith('/base')
    assert type(robot.actuators['base_legs']).__name__ == 'ActuatorNetMLP'
    assert torch.isfinite(env.get_observations()).all()
    assert torch.count_nonzero(env.get_observations()[:, 45:]) == 0
    # Native XYZW orientation and projected gravity agree, including live root state.
    gravity = torch.tensor([0., 0., -1.], device=env.device).expand(args.num_envs, -1)
    torch.testing.assert_close(quat_apply_inverse(robot.data.root_quat_w.torch, gravity), robot.data.projected_gravity_b.torch)
    # Force all rows to timeout. A distinctive action proves final_obs precedes reset.
    env.episode_length_buf[:] = env.max_episode_length - 1
    action = torch.full((args.num_envs, 12), 0.137, device=env.device)
    obs, critic, rewards, done, infos, ids, terminal = env.step(action)
    assert done.all() and len(ids) == args.num_envs
    # Deliberately artificial episode_length_buf must not dilute reward rates.
    # There has been exactly one real step since reset, irrespective of the timeout offset.
    torch.testing.assert_close(sum(infos['him_episode'].values()), rewards.mean()/env.unwrapped.step_dt,
                               atol=1e-4, rtol=1e-4)
    from himloco_lab.utils import export_deploy_cfg as deploy_first
    from himloco_lab.utils import export_deploy_cfg as deploy_second
    assert callable(deploy_first) and deploy_first is deploy_second
    torch.testing.assert_close(terminal[:, 33:45], action[ids])
    torch.testing.assert_close(critic[:, 33:45], torch.zeros_like(action))
    assert torch.count_nonzero(obs[:, 45:]) == 0
    assert torch.count_nonzero(env.unwrapped.pre_pre_action) == 0
    torch.testing.assert_close(critic[:, :45], obs[:, :45])
    assert env.unwrapped.observation_manager.group_obs_dim['critic'] == (238,)
    agent = Go1PPORunnerCfg()
    runner = HIMOnPolicyRunner(env, agent.to_dict(), log_dir=str(output), device=env.device)
    model = runner.alg.actor_critic
    actor_before = model.actor[0].weight.detach().clone()
    estimator_before = model.estimator.encoder[0].weight.detach().clone()
    runner.learn(1)
    assert not torch.equal(actor_before, model.actor[0].weight)
    assert not torch.equal(estimator_before, model.estimator.encoder[0].weight)
    assert all(torch.isfinite(p).all() for p in model.parameters())
    inputs = env.get_observations().clone()
    with torch.inference_mode():
        expected = runner.get_inference_policy()(inputs).clone()
    checkpoint = output / 'model_1.pt'
    loaded = HIMOnPolicyRunner(env, agent.to_dict(), device=env.device)
    loaded.load(str(checkpoint))
    assert loaded.current_learning_iteration == 1
    assert loaded.alg.optimizer.state and loaded.alg.actor_critic.estimator.optimizer.state
    with torch.inference_mode():
        torch.testing.assert_close(loaded.get_inference_policy()(inputs), expected, rtol=0, atol=0)
    exported = output / 'exported'
    export_himloco_policy_as_jit(model, str(exported))
    export_himloco_policy_as_onnx(model, str(exported))
    sample = inputs[:1].cpu()
    with torch.inference_mode():
        reference = model.act_inference(inputs[:1]).cpu()
        jit = torch.jit.load(str(exported / 'policy.pt'))(sample)
    torch.testing.assert_close(jit, reference, atol=2e-5, rtol=2e-5)
    encoder = ort.InferenceSession(str(exported / 'encoder.onnx'), providers=['CPUExecutionProvider'])
    policy = ort.InferenceSession(str(exported / 'policy.onnx'), providers=['CPUExecutionProvider'])
    encoded = encoder.run(None, {'obs_history': sample.numpy()})[0]
    onnx = policy.run(None, {'obs': np.concatenate((sample.numpy()[:, :45], encoded), axis=-1)})[0]
    np.testing.assert_allclose(onnx, reference.numpy(), atol=2e-5, rtol=2e-5)
    result = dict(status='passed', num_envs=args.num_envs, policy=45, history=270, critic=238, action=12,
                  joint_names=robot.joint_names, body_names=robot.body_names,
                  default_joint_pos=robot.data.default_joint_pos.torch[0].tolist(),
                  actuator=type(robot.actuators['base_legs']).__name__,
                  actor_max_delta=(actor_before-model.actor[0].weight).abs().max().item(),
                  estimator_max_delta=(estimator_before-model.estimator.encoder[0].weight).abs().max().item(),
                  jit_max_error=(jit-reference).abs().max().item(),
                  onnx_max_error=float(np.max(np.abs(onnx-reference.numpy()))),
                  timeout_count=int(infos['time_outs'].sum()), unit_tests=suite.countTestCases())
    (output / 'validation.json').write_text(json.dumps(result, indent=2))
    print('GO1_VALIDATION_OK', json.dumps(result))
    env.close()

try:
    main()
except BaseException:
    import traceback
    traceback.print_exc()
    # SimulationApp.close() terminates the process directly; print errors first.
    launcher.app.close(exit_code=1)
else:
    launcher.app.close()
