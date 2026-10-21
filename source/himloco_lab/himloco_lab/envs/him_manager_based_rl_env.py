"""Use Isaac Lab stepping and Same-Step autoreset; only retain action history."""
import torch
from isaaclab.envs import ManagerBasedRLEnv
from himloco_lab.rsl_rl.utils.training_logger import REWARD_ALIASES

class HimlocoManagerBasedRLEnv(ManagerBasedRLEnv):
    def __init__(self, cfg, render_mode=None, **kwargs):
        cfg.compute_final_obs = True
        super().__init__(cfg, render_mode=render_mode, **kwargs)
        self._pre_pre_action = torch.zeros_like(self.action_manager.action)
        self._logging_episode_steps = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

    def step(self, action):
        # Before process_action, prev_action is a[t-2]; action is a[t-1].
        self._logging_episode_steps += 1
        self._pre_pre_action.copy_(self.action_manager.prev_action)
        self.extras.pop("him_episode", None)
        self.extras.pop("final_obs", None)  # Native extras persists between steps.
        self._in_him_step = True
        try:
            return super().step(action)
        finally:
            self._in_him_step = False

    def _reset_idx(self, env_ids):
        # Native UI resets need the same pre-reset data contract as terminations.
        if getattr(self, "_in_him_step", False) and "final_obs" not in self.extras:
            self.extras["final_obs"] = self.observation_manager.compute()
        if getattr(self, "_in_him_step", False) and len(env_ids):
            duration = (self._logging_episode_steps[env_ids].float() * self.step_dt).clamp_min(self.step_dt)
            self.extras["him_episode"] = {
                f"Episode/rew_{REWARD_ALIASES.get(name, name)}": (sums[env_ids] / duration).mean()
                for name, sums in self.reward_manager._episode_sums.items()
                if self.reward_manager.get_term_cfg(name).weight != 0
            }
        super()._reset_idx(env_ids)
        if hasattr(self, "_pre_pre_action"):
            self._pre_pre_action[env_ids] = 0
            self._logging_episode_steps[env_ids] = 0

    @property
    def pre_pre_action(self):
        return self._pre_pre_action
