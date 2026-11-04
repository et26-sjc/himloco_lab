"""CPU preparation and causal indexing; no Isaac Sim imports required."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def file_hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def smoothed_loading_rise(previous, current, lag):
    """[env, physics samples, foot]. Causal boxcar then fixed-time difference.

    previous contains >= 2*lag preceding samples, current a full action interval.
    The value at each sample uses only forces at or before that sample.
    """
    previous, current = np.asarray(previous), np.asarray(current)
    if lag < 1 or previous.shape[1] < 2 * lag:
        raise ValueError('Insufficient within-episode physical history')
    force = np.concatenate((previous[:, -2 * lag:], current), axis=1)
    n = current.shape[1]
    recent = sum(force[:, 2 * lag - i:2 * lag - i + n] for i in range(lag)) / lag
    delayed = sum(force[:, lag - i:lag - i + n] for i in range(lag)) / lag
    return np.maximum(recent - delayed, 0).max(axis=1)


def prepare_collection(root, output, spec):
    """Hash-check original shards, preserve every interval, no random frame split."""
    root, output = Path(root), Path(output)
    report = json.loads((root / 'report.json').read_text())
    if report['status'] != 'completed' or (root / 'INCOMPLETE').exists():
        raise ValueError('Incomplete raw collection')
    if report['scenario'] != 'suite_walk' or report['baseline_observer']:
        raise ValueError('A1 requires the instrumented terrain suite')
    output.mkdir(parents=True, exist_ok=False)
    (output / 'INCOMPLETE').write_text('Dataset preparation not complete\n')
    t, n = report['steps'], report['num_envs']
    shapes = {'pre': ((t, n, 45), 'float32'), 'post': ((t, n, 45), 'float32'),
              'severity': ((t, n, 4), 'float32'), 'age': ((t, n), 'int16'),
              'done': ((t, n), 'bool'), 'terminated': ((t, n), 'bool'),
              'precontact': ((t, n, 4), 'bool'), 'onset_ms': ((t, n, 4), 'float32'),
              'terrain_exposed': ((t, n), 'bool'), 'raw_rise': ((t, n, 4), 'float32'),
              'peak': ((t, n, 4), 'float32'), 'integral': ((t, n, 4), 'float32')}
    arrays = {key: np.lib.format.open_memmap(output / f'{key}.npy', mode='w+', dtype=dtype, shape=shape)
              for key, (shape, dtype) in shapes.items()}
    lag = round(report['window']['rise_interval_s'] / report['window']['physics_dt'])
    dt = report['window']['physics_dt']
    previous = np.zeros((n, 2 * lag, 4), np.float32)
    cursor = 0
    max_offset = 0.0
    for shard in report['shards']:
        path = root / shard['file']
        if file_hash(path) != shard['sha256']:
            raise ValueError(f'Shard hash failed: {path}')
        with np.load(path, allow_pickle=False) as d:
            count = len(d['policy_step'])
            if not np.array_equal(d['policy_step'], np.arange(cursor, cursor + count)):
                raise ValueError('Non-contiguous time')
            sl = slice(cursor, cursor + count)
            for dst, src in [('pre', 'pre_frame'), ('post', 'post_frame'), ('age', 'episode_step'),
                             ('done', 'done'), ('terminated', 'terminated'), ('raw_rise', 'positive_increment_n'),
                             ('peak', 'normal_peak_n'), ('integral', 'normal_integral_ns')]:
                arrays[dst][sl] = d[src]
            xy = d['root_xy_from_origin']
            max_offset = max(max_offset, float(np.abs(xy).max()))
            arrays['terrain_exposed'][sl] = np.abs(xy[..., 0]) > 0.95
            for k in range(count):
                force = np.linalg.norm(d['normal_forces_w'][k], axis=-1)
                current = force[:, lag:]
                # First interval after reset is masked downstream; never bridge it.
                reset = d['episode_step'][k] == 0
                previous[reset] = 0
                arrays['severity'][cursor + k] = smoothed_loading_rise(previous, current, lag)
                arrays['precontact'][cursor + k] = previous[:, -lag:].max(axis=1) < 1.0
                contact = current > 1.0
                onset = (contact.argmax(axis=1) + 1) * dt * 1000
                onset[~contact.any(axis=1)] = np.nan
                arrays['onset_ms'][cursor + k] = onset
                previous = np.concatenate((previous, current), axis=1)[:, -2 * lag:].copy()
            cursor += count
    if cursor != t or max_offset >= 5.5:
        raise ValueError(f'Incomplete time coverage or robot near tile boundary: {cursor}, {max_offset}')
    for a in arrays.values():
        a.flush()
    metadata = {'status': 'completed', 'steps': t, 'num_envs': n, 'collection_seed': report['seed'],
                'collection_report_sha256': file_hash(root / 'report.json'), 'raw_collection': str(root.resolve()),
                'label': spec['label'], 'terrain': report['terrain'], 'max_abs_xy_offset_m': max_offset,
                'physics_dt': dt, 'policy_dt': report['window']['policy_dt'],
                'prepared_sha256': {key: file_hash(output / f'{key}.npy') for key in shapes}}
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    (output / 'INCOMPLETE').unlink()
    return metadata


class CausalDataset:
    """Resident raw frames; reconstruct histories only for each sampled batch."""
    def __init__(self, path, device='cuda:0'):
        self.path = Path(path)
        self.meta = json.loads((self.path / 'metadata.json').read_text())
        if self.meta['status'] != 'completed' or (self.path / 'INCOMPLETE').exists():
            raise ValueError('Incomplete prepared dataset')
        self.device = device
        self.t, self.n = self.meta['steps'], self.meta['num_envs']
        self.cpu = {key: np.load(self.path / f'{key}.npy', mmap_mode='r') for key in
                    ('age', 'severity', 'done', 'terminated', 'precontact', 'onset_ms', 'terrain_exposed')}
        self.family = np.asarray(self.meta['terrain']['family_id'])
        self.tile = np.asarray(self.meta['terrain']['tile_id'])
        self.pre = torch.tensor(np.load(self.path / 'pre.npy', mmap_mode='r'), device=device)
        self.post = torch.tensor(np.load(self.path / 'post.npy', mmap_mode='r'), device=device)
        self.severity = torch.tensor(np.asarray(self.cpu['severity']), device=device)
        self.age = torch.tensor(np.asarray(self.cpu['age']).astype(np.int64), device=device)

    def indices(self, families, role=None):
        env = np.isin(self.family, families)
        if role is not None:
            env &= np.arange(self.n) % 3 == role
        mask = (self.cpu['age'] >= self.meta['label']['ignore_episode_first_steps']) & env[None, :]
        return np.flatnonzero(mask.reshape(-1))

    def batch(self, indices, task, variant='history'):
        ids = torch.as_tensor(indices, dtype=torch.int64, device=self.device)
        time, env = ids // self.n, ids % self.n
        if task not in ('detection', 'forecast'):
            raise ValueError('Unknown causal task')
        if variant == 'single':
            x = self.post[time, env] if task == 'detection' else self.pre[time, env]
        else:
            offsets = torch.arange(5 if task == 'detection' else 6, device=self.device)
            times = (time[:, None] - offsets).clamp_min(0)
            frames = self.pre[times, env[:, None]]
            # This also makes this accessor safe for early-episode diagnostics.
            frames = frames * (offsets[None, :] <= self.age[time, env, None])[..., None]
            if task == 'detection':
                frames = torch.cat((self.post[time, env, None], frames), dim=1)
            if variant == 'shuffled':
                # Independent per-example order; only historical frames shuffled.
                # Keep current frame fixed so this is a chronology ablation.
                order = torch.rand(len(ids), 5, device=self.device).argsort(dim=1) + 1
                frames = torch.cat((frames[:, :1], frames.gather(1, order[..., None].expand(-1, -1, 45))), dim=1)
            x = frames.reshape(len(ids), -1)
        return x, self.severity[time, env]
