# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import time
import os
from collections import deque
import statistics

import torch

from ..algorithms import HIMPPO
from ..modules import HIMActorCritic
from ..env import VecEnv
from ..utils.training_logger import TrainingLogger, CURRICULUM_ALIASES


class HIMOnPolicyRunner:

    def __init__(self,
                 env: VecEnv,
                 train_cfg,
                 log_dir=None,
                 device='cpu'):
        """Initialize HimLoco on-policy runner.
        
        Args:
            env: Vectorized environment.
            train_cfg: Training configuration dictionary.
            log_dir: Directory for logging.
            device: Device to run on.
        """
        # Store configuration
        self.cfg = train_cfg  # Full training config (for logging compatibility)
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env
        
        # Determine observation dimensions
        if self.env.num_privileged_obs is not None:
            num_critic_obs = self.env.num_privileged_obs 
        else:
            num_critic_obs = self.env.num_obs
        self.num_actor_obs = self.env.num_obs
        self.num_critic_obs = num_critic_obs
        
        # Initialize policy network
        actor_critic_class = eval(train_cfg["policy_class_name"])  # HIMActorCritic
        actor_critic: HIMActorCritic = actor_critic_class(
            self.env.num_obs,  # historical obs
            num_critic_obs,  # historical privileged obs
            self.env.num_one_step_obs,
            self.env.num_actions,
            **self.policy_cfg
        ).to(self.device)
        
        # Initialize algorithm
        alg_class = eval(train_cfg["algorithm_class_name"])  # HIMPPO
        self.alg: HIMPPO = alg_class(actor_critic, device=self.device, **self.alg_cfg)
        
        # Training configuration
        self.num_steps_per_env = train_cfg["num_steps_per_env"]
        self.save_interval = train_cfg["save_interval"]

        # Initialize storage and model
        self.alg.init_storage(
            self.env.num_envs, 
            self.num_steps_per_env, 
            [self.env.num_obs], 
            [self.env.num_privileged_obs], 
            [self.env.num_actions]
        )

        # Logging
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.logger_type = self.cfg["logger"].lower()
        self.wandb_run_id = None
        self.counter_source = "checkpoint"

        # _, _ = self.env.reset() we call this in wrapper
    
    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        """Train the policy using HimLoco PPO algorithm.
        
        Args:
            num_learning_iterations: Number of policy updates.
            init_at_random_ep_len: Whether to randomize initial episode lengths.
        """
        # Initialize writer (support multiple logger types like RSL-RL)
        if self.log_dir is not None and self.writer is None:
            # Launch either Tensorboard, WandB, or Neptune summary writer(s)
            if self.logger_type == "neptune":
                from rsl_rl.utils.neptune_utils import NeptuneSummaryWriter
                self.writer = NeptuneSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.alg_cfg)
                # TODO: Add config logging support
                # self.writer.log_config(self.env.cfg, train_cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type in ("wandb", "tensorboard"):
                self.writer = TrainingLogger(self.log_dir, {**self.cfg, 'num_envs': self.env.num_envs,
                    'num_actor_obs': self.num_actor_obs, 'num_critic_obs': self.num_critic_obs,
                    'num_one_step_obs': self.env.num_one_step_obs, 'num_actions': self.env.num_actions,
                    'device': str(self.device)}, self.wandb_run_id)
                if self.writer.run is not None:
                    self.wandb_run_id = self.writer.run.id
            else:
                raise ValueError("Logger type not found. Please choose 'neptune', 'wandb' or 'tensorboard'.")
        
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))
        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        self.alg.actor_critic.train() # switch to train mode (for dropout for example)

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        for it in range(start_iter, tot_iter):
            start = time.time()
            histogram_data = {'Distributions/actions': [], 'Distributions/values': []}
            collect_histograms = (self.cfg.get('wandb_log_histograms', False)
                and (it + 1) % self.cfg.get('wandb_histogram_interval', 10) == 0)
            histogram_budget = self.cfg.get('wandb_histogram_samples', 2048)
            histogram_stride = 10
            per_sample_budget = max(1, histogram_budget // max(1, (self.num_steps_per_env + 9) // 10))
            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, critic_obs)
                    if collect_histograms and i % histogram_stride == 0:
                        # Bound GPU buffers first; transfer only the final small sample once per iteration.
                        histogram_data['Distributions/actions'].append(actions.flatten()[:per_sample_budget].detach())
                        histogram_data['Distributions/values'].append(self.alg.transition.values.flatten()[:per_sample_budget].detach())
                    obs, privileged_obs, rewards, dones, infos, termination_ids, termination_privileged_obs = self.env.step(actions)

                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    obs, critic_obs, rewards, dones = obs.to(self.device), critic_obs.to(self.device), rewards.to(self.device), dones.to(self.device)
                    termination_ids = termination_ids.to(self.device)
                    termination_privileged_obs = termination_privileged_obs.to(self.device)

                    next_critic_obs = critic_obs.clone().detach()
                    next_critic_obs[termination_ids] = termination_privileged_obs.clone().detach()

                    self.alg.process_env_step(rewards, dones, infos, next_critic_obs)
                    # print("+++++++++++++++++++++++++")
                    # print(obs[0])
                    # print(next_critic_obs[0, :self.env.num_one_step_obs])
                    # print("+++++++++++++++++++++++++")
                    if self.log_dir is not None:
                        # Book keeping
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        elif 'log' in infos:
                            ep_infos.append(infos['log'])
                        if 'him_episode' in infos:
                            ep_infos.append(infos['him_episode'])
                        # Update rewards
                        cur_reward_sum += rewards
                        # Update episode length
                        cur_episode_length += 1
                        # Clear data for completed episodes
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(critic_obs)
                
            mean_value_loss, mean_surrogate_loss, mean_estimation_loss, mean_swap_loss = self.alg.update()
            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it + 1
            self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
            self.tot_time += collection_time + learn_time
            # log info
            if self.log_dir is not None:
                self.log(locals())
                if hasattr(self.writer, 'end_iteration'):
                    self.writer.end_iteration(self.current_learning_iteration,
                                              histogram_data if collect_histograms else None)
                # Save model
                if self.current_learning_iteration % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))
            # Clear episode infos
            ep_infos.clear()
        
        if self.writer is not None:
            self.writer.flush()
        # Save the final model after training
        if self.log_dir is not None:
            self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))
        if self.writer is not None and hasattr(self.writer, 'close'):
            self.writer.close()

    def log(self, locs, width=80, pad=35):
        """Log training information to console and TensorBoard.
        
        Enhanced version compatible with RSL-RL style logging, supporting:
        - Dynamic episode info processing
        - Loss dictionary iteration
        - Detailed performance metrics
        """
        iteration_time = locs['collection_time'] + locs['learn_time']

        # -- Episode info 
        ep_string = ""
        if locs['ep_infos']:
            for key in sorted(set().union(*(info.keys() for info in locs['ep_infos']))):
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    # skip missing keys
                    if key not in ep_info:
                        continue
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                
                if infotensor.numel() > 0:
                    value = torch.mean(infotensor)
                    # log to tensorboard with proper namespace
                    if "/" in key:
                        self.writer.add_scalar(key, value, self.current_learning_iteration)
                        if key in CURRICULUM_ALIASES:
                            self.writer.add_scalar(CURRICULUM_ALIASES[key], value, self.current_learning_iteration)
                        ep_string += f"""{f'{key}:':>{pad}} {value:.4f}\n"""
                    else:
                        self.writer.add_scalar('Episode/' + key, value, self.current_learning_iteration)
                        ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        
        # -- Policy metrics
        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

        # -- Losses 
        loss_dict = {
            'value_function': locs['mean_value_loss'],
            'surrogate': locs['mean_surrogate_loss'],
            'Estimation Loss': locs['mean_estimation_loss'],
            'Swap Loss': locs['mean_swap_loss'],
        }
        for key, value in loss_dict.items():
            self.writer.add_scalar(f'Loss/{key}', value, self.current_learning_iteration)
        
        self.writer.add_scalar('Loss/learning_rate', self.alg.learning_rate, self.current_learning_iteration)
        self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), self.current_learning_iteration)
        
        # -- Performance
        self.writer.add_scalar('Perf/total_fps', fps, self.current_learning_iteration)
        self.writer.add_scalar('Perf/collection_time', locs['collection_time'], self.current_learning_iteration)
        self.writer.add_scalar('Perf/learning_time', locs['learn_time'], self.current_learning_iteration)
        
        # -- Training metrics
        if len(locs['rewbuffer']) > 0:
            self.writer.add_scalar('Train/mean_reward', statistics.mean(locs['rewbuffer']), self.current_learning_iteration)
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), self.current_learning_iteration)
            # W&B uses Timing/total_time as the explicit x-axis for these metrics.
            self.writer.add_scalar('Train/mean_reward/time', statistics.mean(locs['rewbuffer']), self.tot_time)
            self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)

        for key, value in {
            'trainer/iteration': self.current_learning_iteration,
            'trainer/timesteps': self.tot_timesteps, 'Metrics/total_timesteps': self.tot_timesteps,
            'Timing/total_time': self.tot_time, 'Timing/iteration_time': iteration_time,
        }.items():
            self.writer.add_scalar(key, value, self.current_learning_iteration)

        # -- Console output
        str_header = f" \033[1m Learning iteration {locs['it']}/{locs['tot_iter']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str_header.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs['collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
            )
            # Add losses
            for key, value in loss_dict.items():
                log_string += f"""{f'Mean {key} loss:':>{pad}} {value:.4f}\n"""
            # Add rewards
            log_string += (
                f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
            )
        else:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str_header.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs['collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
            )
            for key, value in loss_dict.items():
                log_string += f"""{f'{key}:':>{pad}} {value:.4f}\n"""

        # Add episode info
        log_string += ep_string
        
        # Add summary
        log_string += (
            f"""{'-' * width}\n"""
            f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
            f"""{'Time elapsed:':>{pad}} {time.strftime("%H:%M:%S", time.gmtime(self.tot_time))}\n"""
            f"""{'ETA:':>{pad}} {time.strftime(
                "%H:%M:%S", 
                time.gmtime(
                    self.tot_time / (locs['it'] - locs['start_iter'] + 1)
                    * (locs['start_iter'] + locs['num_learning_iterations'] - locs['it'])
                )
            )}\n"""
        )
        print(log_string)

    def save(self, path, infos=None):
        """Save model checkpoint.
        
        Args:
            path: Path to save the model.
            infos: Additional information to save.
        """
        saved_dict = {
            'model_state_dict': self.alg.actor_critic.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'estimator_optimizer_state_dict': self.alg.actor_critic.estimator.optimizer.state_dict(),
            'iter': self.current_learning_iteration,
            'infos': infos,
            'checkpoint_format_version': 2,
            'tot_timesteps': self.tot_timesteps,
            'tot_time': self.tot_time,
            'wandb_run_id': self.wandb_run_id,
        }
        torch.save(saved_dict, path)
        
        # Upload model to external logging service 
        if hasattr(self, 'logger_type') and self.logger_type in ["neptune", "wandb"]:
            if hasattr(self.writer, 'save_model'):
                self.writer.save_model(path, self.current_learning_iteration)

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path, map_location=self.device, weights_only=True)
        self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
            self.alg.actor_critic.estimator.optimizer.load_state_dict(loaded_dict['estimator_optimizer_state_dict'])
        self.alg.learning_rate = self.alg.optimizer.param_groups[0]["lr"]
        self.alg.actor_critic.estimator.learning_rate = self.alg.actor_critic.estimator.optimizer.param_groups[0]["lr"]
        self.current_learning_iteration = loaded_dict['iter']
        self.tot_time = loaded_dict.get('tot_time', 0.)
        self.wandb_run_id = loaded_dict.get('wandb_run_id')
        if 'tot_timesteps' in loaded_dict:
            self.tot_timesteps = int(loaded_dict['tot_timesteps'])
        else:
            # First migration checkpoints did not save counters. Read the actual source run's
            # dimensions, rather than guessing with the current (possibly changed) env count.
            import yaml
            from pathlib import Path
            params = Path(path).parent / 'params'
            if (params/'env.yaml').exists() and (params/'agent.yaml').exists():
                with (params/'env.yaml').open() as stream:
                    env_cfg = yaml.load(stream, Loader=yaml.BaseLoader)
                with (params/'agent.yaml').open() as stream:
                    agent_cfg = yaml.load(stream, Loader=yaml.BaseLoader)
                self.tot_timesteps = (self.current_learning_iteration * int(env_cfg['scene']['num_envs'])
                                      * int(agent_cfg['num_steps_per_env']))
                self.counter_source = 'reconstructed from source params and completed iteration count'
                print(f'[INFO] Reconstructed cumulative timesteps={self.tot_timesteps} from source params; '
                      'assumes the historical env/rollout dimensions stayed constant. Elapsed time is unavailable.')
            elif load_optimizer:
                raise ValueError('Checkpoint lacks cumulative timesteps and source params; cannot resume accurate logs')
        return loaded_dict['infos']

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference
