"""Unthresholded contact proxies with explicit physical-time windows.

These are solver-derived supervision candidates, not validated physical impact
ground truth. No event threshold is selected by this module.
"""
from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class ContactWindow:
    physics_dt: float = 0.005
    policy_dt: float = 0.020
    rise_interval_s: float = 0.010

    def __post_init__(self):
        for name in ("physics_dt", "policy_dt", "rise_interval_s"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("policy_dt", "rise_interval_s"):
            ratio = getattr(self, name) / self.physics_dt
            if not math.isclose(ratio, round(ratio), abs_tol=1e-8, rel_tol=0):
                raise ValueError(f"{name} must be an integer multiple of physics_dt")
            if round(ratio) < 1:
                raise ValueError(f"{name} must contain at least one physics sample")

    @property
    def decimation(self):
        return round(self.policy_dt / self.physics_dt)

    @property
    def rise_steps(self):
        return round(self.rise_interval_s / self.physics_dt)

    @property
    def history_length(self):
        # Four new samples + two preceding samples for a 10 ms difference.
        return self.decimation + self.rise_steps

    @property
    def offsets_s(self):
        """Sample times relative to the start of the action interval [s]."""
        return tuple((i - self.rise_steps + 1) * self.physics_dt for i in range(self.history_length))


def contact_window_targets(normal_forces_w: torch.Tensor, window: ContactWindow):
    """Summarize oldest-first [batch, history, foot, xyz] normal resultants.

    Force and rise targets have units N; loading rate N/s; integral N s.
    The integral includes ordinary support loading. The resultant excludes
    friction and can hide opposing simultaneous contact normals.
    """
    if normal_forces_w.ndim != 4 or normal_forces_w.shape[1:] != (window.history_length, 4, 3):
        raise ValueError("Expected [batch, history_length, 4, 3] normal-force vectors")
    if not torch.isfinite(normal_forces_w).all():
        raise ValueError("Non-finite contact supervision")
    force = torch.linalg.vector_norm(normal_forces_w, dim=-1)
    current = force[:, window.rise_steps:]
    delayed = force[:, :window.decimation]
    increments = (current - delayed).clamp_min(0)
    previous = force[:, window.rise_steps - 1:-1]
    return {
        "normal_peak_n": current.amax(dim=1),
        "normal_integral_ns": current.sum(dim=1) * window.physics_dt,
        "positive_increment_n": increments.amax(dim=1),
        "positive_rate_fixed_window_n_per_s": increments.amax(dim=1) / window.rise_interval_s,
        "positive_rate_one_step_n_per_s": ((current - previous).clamp_min(0) / window.physics_dt).amax(dim=1),
    }


def completed_interval_frame(returned_history, terminal_ids, final_policy_frame, frame_dim=45):
    """Use pre-reset terminal observations, never a new episode's first frame.

    Combine this frame with pre-action history to train detection. Pre-action
    history with this interval's target trains forecasting. They are different
    tasks even when they share the same target tensor.
    """
    if returned_history.ndim != 2 or returned_history.shape[1] != frame_dim * 6:
        raise ValueError("Expected a six-frame returned history")
    frame = returned_history[:, :frame_dim].clone()
    if len(terminal_ids):
        if final_policy_frame is None or final_policy_frame.shape != frame.shape:
            raise ValueError("Missing full-batch pre-reset policy observations")
        frame[terminal_ids] = final_policy_frame[terminal_ids]
    if not torch.isfinite(frame).all():
        raise ValueError("Non-finite completed-interval observations")
    return frame
