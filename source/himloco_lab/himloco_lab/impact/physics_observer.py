"""Read-only substep observer. Import after AppLauncher; no observation RNG calls."""
import torch

from isaaclab.utils.math import quat_apply_inverse


class PhysicsObserver:
    def __init__(self, env):
        self.env = env
        self.active = False
        self.frames = []
        self.original_update = env.scene.update
        env.scene.update = self.update

    def update(self, dt):
        self.original_update(dt)
        if not self.active:
            return
        env = self.env
        robot = env.scene['robot']
        # Call raw term functions only: no corruption, history update or RNG.
        terms = env.cfg.observations.policy
        parts = []
        for name in ('velocity_commands', 'base_ang_vel', 'projected_gravity',
                     'joint_pos_rel', 'joint_vel_rel', 'last_action'):
            term = getattr(terms, name)
            value = term.func(env, **term.params).clone()
            if term.clip is not None:
                value.clamp_(*term.clip)
            if term.scale is not None:
                value *= term.scale
            parts.append(value)
        clean = torch.cat(parts, -1)
        if clean.shape[1] != 45:
            raise RuntimeError('Unexpected clean proprioceptive frame')
        acceleration = robot.data.body_com_lin_acc_w.torch[:, 0]
        gravity = torch.tensor([0., 0., -9.81], device=env.device)
        specific_force = quat_apply_inverse(robot.data.root_link_quat_w.torch, acceleration - gravity)
        effort = robot.data.applied_torque.torch
        self.frames.append(torch.cat((clean, specific_force, effort), -1).clone())

    def begin(self):
        self.frames.clear()
        self.active = True

    def end(self):
        self.active = False
        if len(self.frames) != self.env.cfg.decimation:
            raise RuntimeError(f'Lost physics samples: {len(self.frames)}')
        return torch.stack(self.frames, dim=1)
