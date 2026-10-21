"""Use Isaac Lab stepping and Same-Step autoreset; only retain action history."""
import torch
from isaaclab.envs import ManagerBasedRLEnv

class HimlocoManagerBasedRLEnv(ManagerBasedRLEnv):
    def __init__(self, cfg, render_mode=None, **kwargs):
        cfg.compute_final_obs = True
        super().__init__(cfg, render_mode=render_mode, **kwargs)
        self._pre_pre_action = torch.zeros_like(self.action_manager.action)

    def step(self, action):
        # Before process_action, prev_action is a[t-2]; action is a[t-1].
        self._pre_pre_action.copy_(self.action_manager.prev_action)
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
        super()._reset_idx(env_ids)
        if hasattr(self, "_pre_pre_action"):
            self._pre_pre_action[env_ids] = 0

    @property
    def pre_pre_action(self):
        return self._pre_pre_action
