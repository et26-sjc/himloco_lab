# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Wrapper to configure Isaac Lab environment for HimLoco RSL-RL."""

from __future__ import annotations

import gymnasium as gym
import torch

from himloco_lab.envs import HimlocoManagerBasedRLEnv
from ..env.vec_env import VecEnv
from .history import stack_frame

class HimlocoVecEnvWrapper(VecEnv):
    """Wraps around Isaac Lab environment for HimLoco RSL-RL.
    
    This wrapper adapts Isaac Lab HimlocoManagerBasedRLEnv to the interface required by 
    HimLoco's custom RSL-RL implementation.
    
    Key features:
    - Converts TensorDict observations to plain tensors
    - Maintains history buffers for observations
    - Tracks terminated environments and their privileged observations
    - Provides 7-value return format required by HimLoco
    
    .. caution::
        This class must be the last wrapper in the wrapper chain.
    
    Args:
        env: The Isaac Lab HimlocoManagerBasedRLEnv environment.
        history_length: Number of historical time steps to stack with current observation.
            0 = only current step, 1 = current + 1 past step, etc.
        privileged_history_length: Number of historical time steps to stack with current privileged observation.
            0 = only current step, 1 = current + 1 past step, etc.
    
    Raises:
        ValueError: When the environment is not an instance of HimlocoManagerBasedRLEnv.
    """

    def __init__(
        self, 
        env: HimlocoManagerBasedRLEnv,
        history_length: int = 0,
        privileged_history_length: int = 0,
    ):
        """Initialize the wrapper.
        
        Args:
            env: Isaac Lab HimlocoManagerBasedRLEnv environment.
            history_length: Number of historical time steps to stack with current observation.
                0 = only current step (num_obs = num_one_step_obs)
                1 = current + 1 past step (num_obs = num_one_step_obs * 2)
                n = current + n past steps (num_obs = num_one_step_obs * (n+1))
            privileged_history_length: Number of historical time steps to stack with current privileged observation.
                Same interpretation as history_length.
        """
        # check that input is valid
        if not isinstance(env.unwrapped, HimlocoManagerBasedRLEnv):
            raise ValueError(
                "The environment must be inherited from HimlocoManagerBasedRLEnv. "
                f"Environment type: {type(env)}"
            )
        
        # check that observation manager exists
        if not hasattr(env.unwrapped, "observation_manager"):
            raise ValueError(
                "The environment must have an observation_manager. "
                "HimlocoManagerBasedRLEnv requires observation_manager to be configured."
            )
        # check that action manager exists
        if not hasattr(env.unwrapped, "action_manager"):
            raise ValueError(
                "The environment must have an action_manager. "
                "HimlocoManagerBasedRLEnv requires action_manager to be configured."
            )
        
        # initialize the wrapper
        self.env = env
        
        # store information required by HimLoco RSL-RL
        self.num_envs = self.unwrapped.num_envs
        self.device = self.unwrapped.device
        self.max_episode_length = self.unwrapped.max_episode_length
        
        # obtain action dimension from action manager
        self.num_actions = self.unwrapped.action_manager.total_action_dim
        
        # obtain single-step observation dimensions from observation manager
        # These are the dimensions of observations returned by the environment at each step
        self.num_one_step_obs = self.unwrapped.observation_manager.group_obs_dim["policy"][0]
        
        # Set history lengths
        self.history_length = history_length
        
        # Calculate total observation dimensions
        # history_length=0 means only current step (num_obs = num_one_step_obs)
        # history_length=1 means current + 1 past step (num_obs = num_one_step_obs * 2)
        self.num_obs = self.num_one_step_obs * (self.history_length + 1)
        
        # History buffer for policy observations
        self.obs_history_buf = torch.zeros(self.num_envs, self.num_obs, device=self.device)
        
        # Buffers for terminated environments (HimLoco-specific)
        self._termination_ids = torch.tensor([], dtype=torch.long, device=self.device)
        
        # Initialize privileged observation related attributes if critic observations exist
        if "critic" in self.unwrapped.observation_manager.group_obs_dim:
            self.num_one_step_privileged_obs = self.unwrapped.observation_manager.group_obs_dim["critic"][0]
            self.privileged_history_length = privileged_history_length
            self.num_privileged_obs = self.num_one_step_privileged_obs * (self.privileged_history_length + 1)
            self.privileged_obs_history_buf = torch.zeros(self.num_envs, self.num_privileged_obs, device=self.device)
            self._termination_privileged_obs = torch.zeros(0, self.num_privileged_obs, device=self.device)
        else:
            self.num_one_step_privileged_obs = None
            self.privileged_history_length = None
            self.num_privileged_obs = None
            self.privileged_obs_history_buf = None
            self._termination_privileged_obs = None
        
        # reset at the start since the HimLoco runner does not call reset
        self.reset()

    def __str__(self):
        """Returns the wrapper name and the :attr:`env` representation string."""
        return f"<{type(self).__name__}{self.env}>"

    def __repr__(self):
        """Returns the string representation of the wrapper."""
        return str(self)

    """
    Properties -- Gym.Wrapper
    """

    @property
    def cfg(self) -> object:
        """Returns the configuration class instance of the environment."""
        return self.unwrapped.cfg

    @property
    def render_mode(self) -> str | None:
        """Returns the :attr:`Env` :attr:`render_mode`."""
        return self.env.render_mode

    @property
    def observation_space(self):
        """Returns the :attr:`Env` :attr:`observation_space`."""
        return self.env.observation_space

    @property
    def action_space(self):
        """Returns the :attr:`Env` :attr:`action_space`."""
        return self.env.action_space

    @classmethod
    def class_name(cls) -> str:
        """Returns the class name of the wrapper."""
        return cls.__name__

    @property
    def unwrapped(self):
        """Returns the base environment of the wrapper."""
        return self.env.unwrapped

    """
    Properties
    """

    @property
    def episode_length_buf(self) -> torch.Tensor:
        """The episode length buffer."""
        return self.unwrapped.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor):
        """Set the episode length buffer.
        
        Note:
            This is needed to perform random initialization of episode lengths in HimLoco RSL-RL.
        """
        self.unwrapped.episode_length_buf = value

    """
    Operations - MDP
    """

    def seed(self, seed: int = -1) -> int:
        """Set the seed for the environment."""
        return self.env.seed(seed)

    def _validate(self, obs, privileged):
        for name, tensor, dim in (("policy", obs, self.num_obs),
                                  ("critic", privileged, self.num_privileged_obs)):
            if tensor is not None and (tensor.shape != (self.num_envs, dim) or not torch.isfinite(tensor).all()):
                raise ValueError(f"Invalid {name} observations: {tensor.shape}")

    def reset(self):
        observations, _ = self.env.reset()
        self.obs_history_buf.zero_()
        self.obs_history_buf[:, :self.num_one_step_obs] = observations["policy"]
        if self.privileged_obs_history_buf is not None:
            self.privileged_obs_history_buf.zero_()
            self.privileged_obs_history_buf[:, :self.num_one_step_privileged_obs] = observations["critic"]
        self._validate(self.obs_history_buf, self.privileged_obs_history_buf)
        return self.obs_history_buf, self.privileged_obs_history_buf

    def get_observations(self):
        return self.obs_history_buf

    def get_privileged_observations(self):
        return self.privileged_obs_history_buf

    def step(self, actions):
        if actions.shape != (self.num_envs, self.num_actions) or not torch.isfinite(actions).all():
            raise ValueError("Invalid actions")
        obs, rewards, terminated, truncated, extras = self.env.step(actions)
        dones = terminated | truncated
        ids = dones.nonzero(as_tuple=False).flatten()
        infos = dict(extras)
        # Physical failures take precedence if termination and truncation coincide.
        infos["time_outs"] = (truncated & ~terminated if not self.cfg.is_finite_horizon
                              else torch.zeros_like(truncated))
        terminal = torch.empty((0, self.num_privileged_obs), device=self.device)
        if len(ids):
            if "final_obs" not in extras:
                raise RuntimeError("compute_final_obs must provide pre-reset terminal observations")
            final = extras["final_obs"]["critic"]
            terminal = stack_frame(final, self.privileged_obs_history_buf,
                                   self.privileged_history_length)[ids].clone()
            if not torch.isfinite(terminal).all():
                raise ValueError("Non-finite terminal critic observations")
        # Allocate new buffers: rollout storage still owns references to the old state.
        self.obs_history_buf = stack_frame(obs["policy"], self.obs_history_buf, self.history_length, ids)
        self.privileged_obs_history_buf = stack_frame(
            obs["critic"], self.privileged_obs_history_buf, self.privileged_history_length, ids)
        self._validate(self.obs_history_buf, self.privileged_obs_history_buf)
        if rewards.shape != (self.num_envs,) or not torch.isfinite(rewards).all():
            raise ValueError("Invalid rewards")
        return (self.obs_history_buf, self.privileged_obs_history_buf, rewards,
                dones, infos, ids, terminal)

    def close(self):
        """Close the environment."""
        return self.env.close()

    def __getattr__(self, name):
        """Forward all other attribute access to wrapped environment."""
        return getattr(self.env, name)
