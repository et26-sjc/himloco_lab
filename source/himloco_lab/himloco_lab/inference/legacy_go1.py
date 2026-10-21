"""Semantic adapter for the original Isaac Gym Go1 policy.

DOF order was queried with gym.get_asset_dof_names using the training asset options;
see audit_legacy_asset.py and artifacts/round2/legacy_asset_order.json.
"""
import torch

LEGACY_JOINT_NAMES = [f'{leg}_{joint}_joint' for leg in ('FL', 'FR', 'RL', 'RR')
                      for joint in ('hip', 'thigh', 'calf')]
LEGACY_DEFAULT_POSITIONS = {
    f'{leg}_{joint}_joint': value
    for leg in ('FL', 'FR', 'RL', 'RR')
    for joint, value in [('hip', .1 if leg.endswith('L') else -.1),
                         ('thigh', .8 if leg.startswith('F') else 1.), ('calf', -1.5)]
}


class LegacyGo1ObservationAdapter:
    def __init__(self, observation_joint_names, action_joint_names, lab_default_positions,
                 lab_action_scale=.25, legacy_default_positions=None, device='cpu'):
        if set(observation_joint_names) != set(LEGACY_JOINT_NAMES) or set(action_joint_names) != set(LEGACY_JOINT_NAMES):
            raise ValueError('Go1 adapter requires exactly the 12 named legacy joints')
        if len(set(observation_joint_names)) != 12 or len(set(action_joint_names)) != 12:
            raise ValueError('Duplicate joint names')
        self.obs_to_old = torch.tensor([observation_joint_names.index(n) for n in LEGACY_JOINT_NAMES], device=device)
        self.action_to_old = torch.tensor([action_joint_names.index(n) for n in LEGACY_JOINT_NAMES], device=device)
        self.old_to_action = torch.tensor([LEGACY_JOINT_NAMES.index(n) for n in action_joint_names], device=device)
        self.lab_default_obs = torch.tensor([lab_default_positions[n] for n in observation_joint_names], device=device)
        self.lab_default_action = torch.tensor([lab_default_positions[n] for n in action_joint_names], device=device)
        defaults = legacy_default_positions or LEGACY_DEFAULT_POSITIONS
        self.old_default = torch.tensor([defaults[n] for n in LEGACY_JOINT_NAMES], device=device)
        self.lab_action_scale = lab_action_scale
        self.command_scale = torch.tensor([2., 2., .25], device=device)

    def adapt_frame(self, frame):
        if frame.ndim != 2 or frame.shape[-1] != 45 or not torch.isfinite(frame).all():
            raise ValueError('Expected finite [batch,45] Lab frames')
        result = frame.clone()
        result[:, :3] = frame[:, :3] * self.command_scale
        absolute_pos = frame[:, 9:21] + self.lab_default_obs
        result[:, 9:21] = absolute_pos[:, self.obs_to_old] - self.old_default
        result[:, 21:33] = frame[:, 21:33][:, self.obs_to_old]
        # Recover the legacy raw action from the previously commanded absolute target.
        previous_target = self.lab_default_action + self.lab_action_scale * frame[:, 33:45]
        result[:, 33:45] = (previous_target[:, self.action_to_old] - self.old_default) / .25
        return result

    def adapt_history(self, history, reset_ids=None):
        if history.ndim != 2 or history.shape[-1] != 270:
            raise ValueError('Expected [batch,270] history, current frame first')
        frames = history.reshape(-1, 6, 45)
        # A real frame contains a unit gravity vector; all-zero frames are reset padding.
        valid = frames.ne(0).any(dim=-1)
        result = self.adapt_frame(frames.reshape(-1, 45)).reshape(-1, 6, 45)
        result[~valid] = 0
        if reset_ids is not None:
            result[reset_ids, 0, 33:45] = 0
        return result.flatten(1)

    def to_lab_action(self, old_action):
        if old_action.ndim != 2 or old_action.shape[-1] != 12 or not torch.isfinite(old_action).all():
            raise ValueError('Expected finite [batch,12] legacy action')
        # Preserve the old position target even if the Lab default posture differs.
        old_target = self.old_default + .25 * old_action.clamp(-100., 100.)
        return (old_target[:, self.old_to_action] - self.lab_default_action) / self.lab_action_scale
