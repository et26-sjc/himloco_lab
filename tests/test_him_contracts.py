"""CPU data contracts; also run from validate_go1.py after AppLauncher."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from himloco_lab.rsl_rl.modules.him_actor_critic import HIMActorCritic
from himloco_lab.rsl_rl.algorithms.him_ppo import HIMPPO
from himloco_lab.rsl_rl.wrappers.history import stack_frame
from himloco_lab.rsl_rl.wrappers.himloco_vec_env_wrapper import HimlocoVecEnvWrapper
from isaaclab.utils.math import quat_apply_inverse, quat_from_euler_xyz


class HIMContracts(unittest.TestCase):
    def policy(self):
        return HIMActorCritic(270, 238, 45, 12, actor_hidden_dims=[32, 16], critic_hidden_dims=[32, 16])

    def test_history_order_and_episode_reset(self):
        past = torch.zeros(2, 6)
        for t in (1, 2, 3):
            past = stack_frame(torch.full((2, 2), float(t)), past, 2)
        torch.testing.assert_close(past[0], torch.tensor([3., 3., 2., 2., 1., 1.]))
        saved = past.clone()
        out = stack_frame(torch.full((2, 2), 4.), past, 2, torch.tensor([0]))
        torch.testing.assert_close(past, saved)  # rollout state's reference is immutable
        torch.testing.assert_close(out[0], torch.tensor([4., 4., 0., 0., 0., 0.]))
        torch.testing.assert_close(out[1], torch.tensor([4., 4., 3., 3., 2., 2.]))

    def test_xyzw_gravity_and_yaw(self):
        g = torch.tensor([[0., 0., -1.]])
        torch.testing.assert_close(quat_apply_inverse(torch.tensor([[0., 0., 0., 1.]]), g), g)
        q = quat_from_euler_xyz(torch.zeros(1), torch.tensor([torch.pi / 2]), torch.zeros(1))
        torch.testing.assert_close(quat_apply_inverse(q, g), torch.tensor([[1., 0., 0.]]), atol=1e-6, rtol=0)
        q = quat_from_euler_xyz(torch.zeros(1), torch.zeros(1), torch.tensor([torch.pi / 2]))
        torch.testing.assert_close(quat_apply_inverse(q, torch.tensor([[1., 0., 0.]])),
                                   torch.tensor([[0., -1., 0.]]), atol=1e-6, rtol=0)

    def test_actor_detaches_estimator(self):
        model = self.policy()
        obs = torch.randn(8, 270)
        vel, latent = model.estimator(obs)
        self.assertEqual(vel.shape, (8, 3)); self.assertEqual(latent.shape, (8, 16))
        model.act(obs)
        model.action_mean.square().mean().backward()
        self.assertTrue(all(p.grad is None for p in model.estimator.parameters()))
        self.assertTrue(any(p.grad is not None for p in model.actor.parameters()))
        self.assertEqual(model.act_inference(obs).shape, (8, 12))

    def test_estimator_target_slice_and_update(self):
        model = self.policy()
        estimator = model.estimator
        obs = torch.randn(16, 270)
        critic = torch.randn(16, 238)
        captured = []
        hook = estimator.target.register_forward_pre_hook(lambda _, inputs: captured.append(inputs[0].clone()))
        original_mse = torch.nn.functional.mse_loss
        targets = []
        def mse(pred, target):
            targets.append(target.clone()); return original_mse(pred, target)
        before = estimator.encoder[0].weight.clone()
        with patch('torch.nn.functional.mse_loss', mse):
            losses = estimator.update(obs, critic)
        hook.remove()
        torch.testing.assert_close(captured[0], critic[:, 3:48])
        torch.testing.assert_close(targets[0], critic[:, 45:48])
        self.assertFalse(torch.equal(before, estimator.encoder[0].weight))
        self.assertTrue(all(torch.isfinite(torch.tensor(losses))))

    def test_timeout_bootstrap_uses_terminal_value(self):
        model = self.policy(); ppo = HIMPPO(model, gamma=0.99)
        ppo.init_storage(2, 1, [270], [238], [12])
        obs = torch.randn(2, 270); critic = torch.randn(2, 238)
        ppo.act(obs, critic)
        terminal = torch.randn(2, 238) + 7
        expected = torch.ones(2)
        expected[0] += 0.99 * model.evaluate(terminal[:1]).detach().squeeze()
        ppo.process_env_step(torch.ones(2), torch.ones(2, dtype=torch.bool),
                             {'time_outs': torch.tensor([True, False])}, terminal)
        torch.testing.assert_close(ppo.storage.rewards[0, :, 0], expected)
        torch.testing.assert_close(ppo.storage.next_privileged_observations[0], terminal)

    def test_ppo_and_estimator_update_and_mean_loss(self):
        model = self.policy(); ppo = HIMPPO(model, num_learning_epochs=2, num_mini_batches=2)
        ppo.init_storage(8, 4, [270], [238], [12])
        with torch.inference_mode():
            for _ in range(4):
                ppo.act(torch.randn(8, 270), torch.randn(8, 238))
                ppo.process_env_step(torch.rand(8), torch.zeros(8, dtype=torch.bool), {}, torch.randn(8, 238))
            ppo.compute_returns(torch.randn(8, 238))
        before = {n: p.clone() for n, p in model.named_parameters()}
        update = model.estimator.update; losses = []
        def recorded(*args, **kwargs):
            result = update(*args, **kwargs); losses.append(result); return result
        with patch.object(model.estimator, 'update', recorded):
            result = ppo.update()
        torch.testing.assert_close(torch.tensor(result[2:]), torch.tensor(losses).mean(0))
        for prefix in ('actor.', 'critic.', 'estimator.encoder.', 'estimator.target.', 'estimator.proto.'):
            self.assertTrue(any(not torch.equal(before[n], p) for n, p in model.named_parameters() if n.startswith(prefix)))
        self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))

    def test_wrapper_mixed_terminations_and_final_history(self):
        wrapper = object.__new__(HimlocoVecEnvWrapper)
        wrapper.num_envs = 3; wrapper.num_actions = 1; wrapper.device = 'cpu'
        wrapper.num_one_step_obs = 2; wrapper.num_obs = 4; wrapper.history_length = 1
        wrapper.num_one_step_privileged_obs = 3; wrapper.num_privileged_obs = 6
        wrapper.privileged_history_length = 1
        wrapper.obs_history_buf = torch.ones(3, 4)
        wrapper.privileged_obs_history_buf = torch.full((3, 6), 2.)
        observations = {'policy': torch.full((3, 2), 3.), 'critic': torch.full((3, 3), 4.)}
        final = {'critic': torch.full((3, 3), 7.)}
        # env0 timeout, env1 timeout + failure, env2 continues.
        step = lambda _: (observations, torch.ones(3), torch.tensor([False, True, False]),
                           torch.tensor([True, True, False]), {'final_obs': final})
        wrapper.env = SimpleNamespace(step=step, cfg=SimpleNamespace(is_finite_horizon=False))
        wrapper.env.unwrapped = wrapper.env
        obs, critic, reward, done, info, ids, terminal = wrapper.step(torch.zeros(3, 1))
        torch.testing.assert_close(ids, torch.tensor([0, 1]))
        torch.testing.assert_close(info['time_outs'], torch.tensor([True, False, False]))
        torch.testing.assert_close(terminal, torch.tensor([[7., 7., 7., 2., 2., 2.]]).expand(2, -1))
        torch.testing.assert_close(obs[0], torch.tensor([3., 3., 0., 0.]))
        torch.testing.assert_close(obs[2], torch.tensor([3., 3., 1., 1.]))
        torch.testing.assert_close(critic[0], torch.tensor([4., 4., 4., 0., 0., 0.]))
        wrapper.env.cfg.is_finite_horizon = True
        self.assertFalse(wrapper.step(torch.zeros(3, 1))[4]['time_outs'].any())
