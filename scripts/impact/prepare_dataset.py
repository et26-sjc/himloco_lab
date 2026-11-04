"""Create immutable causal frame caches and A0 loading-proxy statistics."""
import argparse
import json
from pathlib import Path

import numpy as np

from himloco_lab.impact.dataset import prepare_collection


def statistics(output, meta, spec):
    output = Path(output)
    severity = np.load(output / 'severity.npy', mmap_mode='r')
    age = np.load(output / 'age.npy', mmap_mode='r')
    exposed = np.load(output / 'terrain_exposed.npy', mmap_mode='r')
    terminated = np.load(output / 'terminated.npy', mmap_mode='r')
    families = np.asarray(meta['terrain']['family_id'])
    rows = {}
    for i, name in enumerate(meta['terrain']['families']):
        env = families == i
        valid = age[:, env] >= spec['label']['ignore_episode_first_steps']
        s = severity[:, env][valid]
        raw = np.load(output / 'raw_rise.npy', mmap_mode='r')[:, env][valid]
        peak = np.load(output / 'peak.npy', mmap_mode='r')[:, env][valid]
        rows[name] = {'robots': int(env.sum()), 'tiles': int(len(np.unique(np.asarray(meta['terrain']['tile_id'])[env]))),
            'valid_robot_intervals': int(valid.sum()), 'geometry_exposed_fraction': float(exposed[:, env][valid].mean()),
            'smoothed_loading_quantiles_n': np.quantile(s, [0, .5, .9, .95, .99, .999, 1]).tolist(),
            'raw_rise_quantiles_n': np.quantile(raw, [.5, .9, .99, 1]).tolist(),
            'peak_quantiles_n': np.quantile(peak, [.5, .9, .99, 1]).tolist(),
            'exceedance_fractions': {str(t): float((s > t).mean()) for t in spec['label']['thresholds_n']},
            'terminated_count': int(terminated[:, env].sum())}
    return {'status': 'completed', 'interpretation': spec['label']['interpretation'],
            'metadata_sha256_reference': str(output / 'metadata.json'), 'families': rows,
            'quantile_levels': [0, .5, .9, .95, .99, .999, 1]}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('collection', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--spec', type=Path, default=Path('configs/impact/step1.json'))
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    meta = prepare_collection(args.collection, args.output, spec)
    result = statistics(args.output, meta, spec)
    (args.output / 'label_statistics.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
