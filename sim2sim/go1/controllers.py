"""Joint-side Nm commands. Controller gains are separate from passive damping."""
import numpy as np
from .config import gain_vector


class PDController:
    def __init__(self, cfg, joint_names, effort_limits):
        self.kp = gain_vector(cfg.kp, joint_names)
        self.kd = gain_vector(cfg.kd, joint_names)
        self.effort_limits = np.asarray(effort_limits)
        self.computed_torque = np.zeros(len(joint_names))

    def reset(self):
        self.computed_torque.fill(0)

    def step(self, q_target, q, dq):
        self.computed_torque = self.kp * (q_target - q) - self.kd * dq
        if not np.isfinite(self.computed_torque).all():
            raise FloatingPointError('Non-finite PD torque')
        return np.clip(self.computed_torque, -self.effort_limits, self.effort_limits)
