"""Contact-only observer on native stepping; no recorder or controller changes.

Import after AppLauncher. Terminal traces are captured before scene.reset().
"""
import torch

from himloco_lab.envs import HimlocoManagerBasedRLEnv

from .labels import ContactWindow


FOOT_NAMES = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")


class ImpactRecordingEnv(HimlocoManagerBasedRLEnv):
    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)
        self.impact_window = ContactWindow(cfg.sim.dt, cfg.sim.dt * cfg.decimation)
        sensor = self.scene["impact_contacts"]
        indices, names = sensor.find_bodies(FOOT_NAMES, preserve_order=True)
        if tuple(names) != FOOT_NAMES or len(sensor.body_names) != 4:
            raise ValueError(f"Unexpected impact sensor bodies: {sensor.body_names}")
        if sensor.cfg.history_length != self.impact_window.history_length:
            raise ValueError("Impact sensor history does not cover the entire difference window")
        self._impact_indices = indices
        self._reference_indices, names = self.scene["contact_forces"].find_bodies(FOOT_NAMES, preserve_order=True)
        if tuple(names) != FOOT_NAMES:
            raise ValueError("Reference contact sensor foot order mismatch")
        self.impact_normal_history_w = torch.zeros(
            self.num_envs, self.impact_window.history_length, 4, 3, device=self.device)
        self._impact_terminal = torch.zeros_like(self.impact_normal_history_w)
        self.impact_terminal_captured = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.impact_reference_max_error = 0.0
        self.impact_history_overlap_max_error = 0.0
        self._impact_previous_tail = None
        self._impact_previous_done = None
        self._impact_collecting = False

    def _read_impact_history(self):
        # Native sensor uses newest-first; archive chronological samples.
        history = self.scene["impact_contacts"].data.net_normal_forces_w_history.torch
        history = history[:, :, self._impact_indices].flip(1).clone()
        reference = self.scene["contact_forces"].data.net_normal_forces_w_history.torch
        reference = reference[:, :, self._reference_indices].flip(1)
        count = min(reference.shape[1], history.shape[1])
        error = (reference[:, -count:] - history[:, -count:]).abs().amax().item()
        self.impact_reference_max_error = max(self.impact_reference_max_error, error)
        return history

    def step(self, action):
        self.impact_terminal_captured.zero_()
        self._impact_collecting = True
        try:
            result = super().step(action)
        finally:
            self._impact_collecting = False
        history = self._read_impact_history()
        history[self.impact_terminal_captured] = self._impact_terminal[self.impact_terminal_captured]
        done = result[2] | result[3]
        if not torch.equal(done, self.impact_terminal_captured):
            raise RuntimeError("Terminal contact capture does not match completed episodes")
        if self._impact_previous_tail is not None:
            continuing = ~self._impact_previous_done
            if continuing.any():
                error = (history[continuing, :self.impact_window.rise_steps]
                         - self._impact_previous_tail[continuing]).abs().amax().item()
                self.impact_history_overlap_max_error = max(self.impact_history_overlap_max_error, error)
        self._impact_previous_tail = history[:, -self.impact_window.rise_steps:].clone()
        self._impact_previous_done = done.clone()
        self.impact_normal_history_w = history
        return result

    def _reset_idx(self, env_ids):
        if getattr(self, "_impact_collecting", False) and len(env_ids):
            history = self._read_impact_history()
            self._impact_terminal[env_ids] = history[env_ids]
            self.impact_terminal_captured[env_ids] = True
        super()._reset_idx(env_ids)
