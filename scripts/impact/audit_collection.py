"""CPU-only shard/timestamp audit; does not select labels or train a model."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def audit(root):
    root = Path(root)
    report = json.loads((root / 'report.json').read_text())
    if report['status'] != 'completed' or (root / 'INCOMPLETE').exists():
        raise ValueError('Collection did not complete')
    num_envs = report['num_envs']
    window = report['window']
    decimation = round(window['policy_dt'] / window['physics_dt'])
    lag = round(window['rise_interval_s'] / window['physics_dt'])
    expected_step = 0
    episode = np.zeros(num_envs, dtype=np.int64)
    age = episode.copy()
    previous_frame = previous_done = previous_tail = None
    max_frame_error = max_overlap_error = 0.0
    target_errors = {}
    force_max = 0.0
    resets = valid_increments = 0
    total_bytes = 0
    for shard in report['shards']:
        path = root / shard['file']
        if sha256(path) != shard['sha256']:
            raise ValueError(f'Shard hash mismatch: {path}')
        total_bytes += path.stat().st_size
        with np.load(path, allow_pickle=False) as data:
            if len(data['policy_step']) != shard['steps']:
                raise ValueError('Shard length mismatch')
            for name in data.files:
                if not np.isfinite(data[name]).all():
                    raise ValueError(f'Non-finite stored array: {name}')
            for i in range(shard['steps']):
                if data['policy_step'][i] != expected_step:
                    raise ValueError('Missing, duplicated or reordered policy interval')
                if not np.array_equal(data['episode_id'][i], episode) or not np.array_equal(data['episode_step'][i], age):
                    raise ValueError('Episode IDs/ages do not match the preceding dones')
                if not np.array_equal(data['done'][i], data['terminated'][i] | data['truncated'][i]):
                    raise ValueError('Termination metadata mismatch')
                valid = age * decimation >= lag
                if not np.array_equal(data['increment_valid'][i], valid):
                    raise ValueError('Reset padding leaked into a valid increment target')
                valid_increments += int(valid.sum())
                if previous_frame is not None:
                    continuing = ~previous_done
                    if continuing.any():
                        error = float(np.abs(data['pre_frame'][i, continuing] - previous_frame[continuing]).max())
                        max_frame_error = max(max_frame_error, error)
                previous_frame = data['post_frame'][i].copy()
                if not report['baseline_observer']:
                    vectors = data['normal_forces_w'][i]
                    if vectors.shape != (num_envs, decimation + lag, 4, 3):
                        raise ValueError('Force history does not cover the declared window')
                    if previous_tail is not None and (~previous_done).any():
                        continuing = ~previous_done
                        error = float(np.abs(vectors[continuing, :lag] - previous_tail[continuing]).max())
                        max_overlap_error = max(max_overlap_error, error)
                    previous_tail = vectors[:, -lag:].copy()
                    # Recompute independently in float64. GPU float32 norm and
                    # subtraction can differ by ulps; a derivative amplifies
                    # those differences. Bound error in force units first.
                    forces = np.linalg.norm(vectors.astype(np.float64), axis=-1)
                    current = forces[:, lag:]
                    increments = np.maximum(current - forces[:, :decimation], 0)
                    targets = {
                        'normal_peak_n': current.max(axis=1),
                        'normal_integral_ns': current.sum(axis=1) * window['physics_dt'],
                        'positive_increment_n': increments.max(axis=1),
                        'positive_rate_fixed_window_n_per_s': increments.max(axis=1) / window['rise_interval_s'],
                        'positive_rate_one_step_n_per_s': np.maximum(
                            current - forces[:, lag - 1:-1], 0).max(axis=1) / window['physics_dt']
                    }
                    force_roundoff = 8 * np.finfo(np.float32).eps * np.maximum(1.0, forces.max(axis=1))
                    factors = {
                        'normal_peak_n': 1.0, 'normal_integral_ns': decimation * window['physics_dt'],
                        'positive_increment_n': 2.0,
                        'positive_rate_fixed_window_n_per_s': 2.0 / window['rise_interval_s'],
                        'positive_rate_one_step_n_per_s': 2.0 / window['physics_dt']
                    }
                    for key, expected in targets.items():
                        error = np.abs(data[key][i] - expected)
                        target_errors[key] = max(target_errors.get(key, 0.0), float(error.max()))
                        if np.any(error > force_roundoff * factors[key]):
                            raise ValueError(f'Candidate formula exceeds float32 roundoff bound: {key}')
                    force_max = max(force_max, float(current.max()))
                previous_done = data['done'][i].copy()
                resets += int(previous_done.sum())
                episode += previous_done
                age += 1
                age[previous_done] = 0
                expected_step += 1
    if expected_step != report['steps'] or resets != report['reset_count']:
        raise ValueError('Report does not describe the entire collection')
    if max_frame_error != 0 or max_overlap_error != 0:
        raise ValueError('Observation/force window continuity is broken')
    return {
        'passed': True, 'root': str(root), 'policy_intervals': expected_step,
        'transitions': expected_step * num_envs, 'resets': resets,
        'valid_increment_rows': valid_increments,
        'post_to_next_pre_frame_max_error': max_frame_error,
        'force_history_overlap_max_error': max_overlap_error,
        'numpy64_vs_torch32_target_max_errors': target_errors,
        'numeric_tolerance': '8 float32 eps times per-foot force scale, propagated into each target unit',
        'max_normal_force_n': None if report['baseline_observer'] else force_max,
        'shard_bytes': total_bytes, 'bytes_per_transition': total_bytes / (expected_step * num_envs),
        'scope': 'data integrity and candidate formulas only; not physical label validity or learned prediction quality'
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('collection', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = audit(args.collection)
    if args.output:
        with args.output.open('x') as stream:
            stream.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
