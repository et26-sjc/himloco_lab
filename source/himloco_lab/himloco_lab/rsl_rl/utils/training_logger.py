"""One metrics batch per iteration; local TensorBoard plus optional W&B."""
import json
import os
from pathlib import Path

# Aliases describe the enabled Lab contribution, not numerical equivalence to Gym.
REWARD_ALIASES = {
    'track_lin_vel_xy': 'tracking_lin_vel', 'track_ang_vel_z': 'tracking_ang_vel',
    'base_linear_velocity': 'lin_vel_z', 'base_angular_velocity': 'ang_vel_xy',
    'flat_orientation_l2': 'orientation', 'joint_acc': 'dof_acc', 'energy': 'joint_power',
    'base_height_l2': 'base_height', 'feet_height_body': 'foot_clearance',
    'action_rate': 'action_rate', 'smoothness': 'smoothness',
}
CURRICULUM_ALIASES = {'Curriculum/terrain_levels': 'Episode/terrain_level',
                      'Curriculum/lin_vel_cmd_levels': 'Episode/max_command_x'}


class TrainingLogger:
    def __init__(self, log_dir, cfg, parent_run_id=None):
        from torch.utils.tensorboard import SummaryWriter
        self.tb = SummaryWriter(log_dir=log_dir, flush_secs=10)
        self.cfg = cfg; self.run = None; self.pending = {}; self.closed = False
        self.path = Path(log_dir)
        self.metrics_file = (self.path/'metrics.jsonl').open('a')
        if cfg['logger'] == 'wandb':
            import wandb
            if wandb.run is not None:
                raise RuntimeError('A W&B run is already active; refusing duplicate initialization')
            mode = cfg.get('wandb_mode') or os.getenv('WANDB_MODE', 'online')
            # Offline resume creates another session; the counters and parent ID are preserved.
            run_id = cfg.get('wandb_run_id') or (parent_run_id if mode == 'online' else None)
            self.run = wandb.init(
                project=cfg.get('wandb_project', 'himloco-go1-baseline'), entity=os.getenv('WANDB_ENTITY'),
                name=cfg.get('run_name') or None, group=cfg.get('wandb_group'), tags=cfg.get('wandb_tags') or [],
                mode=mode, dir=log_dir, config={**cfg, 'parent_run_id': parent_run_id},
                id=run_id, resume='allow' if run_id and mode == 'online' else None,
                sync_tensorboard=False)
            self.run.define_metric('trainer/iteration')
            self.run.define_metric('*', step_metric='trainer/iteration')
            for key in ('Train/mean_reward/time', 'Train/mean_episode_length/time'):
                self.run.define_metric(key, step_metric='Timing/total_time')
            (self.path/'wandb_session.json').write_text(json.dumps(
                dict(id=self.run.id, project=self.run.project, mode=mode, dir=self.run.dir,
                     parent_run_id=parent_run_id, sync_tensorboard=False), indent=2))

    def add_scalar(self, tag, scalar, global_step):
        value = float(scalar)
        self.tb.add_scalar(tag, value, global_step)
        self.pending[tag] = value

    def end_iteration(self, iteration, histograms=None):
        payload = dict(self.pending)
        sample_counts = {}
        if histograms:
            import torch
            for key, chunks in histograms.items():
                if not chunks:
                    continue
                samples = torch.cat(chunks).flatten()[:self.cfg.get('wandb_histogram_samples', 2048)].cpu().numpy()
                sample_counts[key] = len(samples)
                self.tb.add_histogram(key, samples, iteration)
                if self.run is not None:
                    import wandb
                    payload[key] = wandb.Histogram(samples)
        self.metrics_file.write(json.dumps({**self.pending, 'histogram_sample_counts': sample_counts})+'\n')
        self.metrics_file.flush()
        if self.run is not None:
            self.run.log(payload, step=iteration)
        self.pending.clear()

    def save_model(self, path, iteration):
        if self.run is not None and self.cfg.get('wandb_upload_checkpoints', False):
            import wandb
            artifact = wandb.Artifact(f'checkpoint-{self.run.id}', type='model', metadata={'iteration': iteration})
            artifact.add_file(path)
            self.run.log_artifact(artifact, aliases=[f'iteration-{iteration}', 'latest'])

    def flush(self):
        self.tb.flush(); self.metrics_file.flush()

    def close(self):
        if self.closed:
            return
        self.flush(); self.tb.close(); self.metrics_file.close()
        if self.run is not None:
            self.run.finish()
        self.closed = True
