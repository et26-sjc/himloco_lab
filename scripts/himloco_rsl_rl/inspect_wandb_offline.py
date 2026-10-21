"""Read actual local W&B history records without network access or API credentials."""
import argparse
import json
from pathlib import Path
import struct
import zlib
from wandb.proto.wandb_internal_pb2 import Record

parser = argparse.ArgumentParser()
parser.add_argument('run_dir')
parser.add_argument('--output', required=True)
parser.add_argument('--max_histogram_samples', type=int, default=2048)
args = parser.parse_args()
files = list(Path(args.run_dir).glob('wandb/offline-run-*/run-*.wandb'))
assert len(files) == 1, files
# W&B 0.31 removed its Python DataStore reader. Read its documented LevelDB-style
# framing directly, including checksums, then use the installed protobuf schema.
# Format: https://github.com/wandb/wandb/blob/v0.19.11/wandb/sdk/internal/datastore.py

def records(path):
    raw = path.read_bytes()
    assert raw[:7] == b':W&B\xe1\xbe\x00', 'Unsupported W&B file header'
    offset = 7; fragments = []
    while offset < len(raw):
        remaining = 32768 - offset % 32768
        if remaining < 7:
            assert raw[offset:offset+remaining] == bytes(remaining)
            offset += remaining
            continue
        checksum, length, kind = struct.unpack_from('<IHB', raw, offset)
        assert 1 <= kind <= 4
        offset += 7
        payload = raw[offset:offset+length]; offset += length
        assert len(payload) == length and zlib.crc32(bytes([kind])+payload) & 0xffffffff == checksum
        if kind == 1:
            assert not fragments
            yield payload
        elif kind == 2:
            assert not fragments
            fragments = [payload]
        else:
            assert fragments
            fragments.append(payload)
            if kind == 4:
                yield b''.join(fragments)
                fragments = []
    assert not fragments

rows = []
run_metadata = {}
for data in records(files[0]):
    record = Record(); record.ParseFromString(data)
    if record.HasField('run'):
        run_metadata = {'id': record.run.run_id, 'group': record.run.run_group, 'tags': list(record.run.tags)}
    if record.HasField('history'):
        row = {}
        for item in record.history.item:
            key = item.key or '/'.join(item.nested_key)
            row[key] = json.loads(item.value_json)
        rows.append(row)
metrics = [r for r in rows if 'trainer/iteration' in r]
assert metrics
required = {'trainer/iteration', 'trainer/timesteps', 'Metrics/total_timesteps', 'Loss/Estimation Loss',
            'Loss/Swap Loss', 'Loss/value_function', 'Loss/surrogate', 'Loss/learning_rate',
            'Policy/mean_noise_std', 'Train/mean_reward', 'Train/mean_episode_length',
            'Perf/total_fps', 'Perf/collection_time', 'Perf/learning_time',
            'Timing/total_time', 'Timing/iteration_time', 'Episode/terrain_level', 'Episode/max_command_x'}
required |= {f'Episode/rew_{name}' for name in ['tracking_lin_vel', 'tracking_ang_vel', 'lin_vel_z',
             'ang_vel_xy', 'orientation', 'dof_acc', 'joint_power', 'base_height', 'foot_clearance',
             'action_rate', 'smoothness']}
for row in metrics:
    assert required <= row.keys(), required-row.keys()
    assert row['trainer/timesteps'] == row['Metrics/total_timesteps']
    assert int(row['_step']) == int(row['trainer/iteration'])
    for key in ('Distributions/actions/values', 'Distributions/values/values'):
        if key in row:
            assert sum(row[key]) <= args.max_histogram_samples
iterations = [int(r['trainer/iteration']) for r in metrics]
steps = [int(r['trainer/timesteps']) for r in metrics]
assert all(b > a for a,b in zip(iterations, iterations[1:]))
assert all(b > a for a,b in zip(steps, steps[1:]))
report = dict(path=str(files[0]), history_rows=len(metrics), iterations=iterations, timesteps=steps,
              scalar_keys=sorted(k for k,v in metrics[-1].items() if isinstance(v,(int,float))),
              histogram_keys=sorted(k for k in metrics[-1] if k.startswith('Distributions/')),
              estimator_losses=[r['Loss/Estimation Loss'] for r in metrics],
              swap_losses=[r['Loss/Swap Loss'] for r in metrics],
              run_metadata=run_metadata,
              histogram_sample_counts={key: sum(metrics[-1][key+'/values'])
                  for key in ('Distributions/actions', 'Distributions/values') if key+'/values' in metrics[-1]},
              status='passed', all_required_metrics_present=True)
Path(args.output).write_text(json.dumps(report, indent=2))
print('WANDB_OFFLINE_VALIDATION_OK', json.dumps(report))
