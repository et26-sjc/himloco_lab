"""Reject data corruption and episode leakage before training can use shards."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np


spec = importlib.util.spec_from_file_location(
    'impact_data_audit', Path(__file__).resolve().parents[1] / 'scripts/impact/audit_collection.py')
audit_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit_module)


class ImpactDataAuditTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        # Episode 0 spans steps 0,1. Step 2 begins episode 1; terminal
        # post_frame=2 must not be compared with the reset pre_frame=0.
        self.data = {
            'policy_step': np.arange(3), 'episode_id': np.array([[0], [0], [1]]),
            'episode_step': np.array([[0], [1], [0]]),
            'done': np.array([[False], [True], [False]]),
            'terminated': np.zeros((3, 1), dtype=bool),
            'truncated': np.array([[False], [True], [False]]),
            'increment_valid': np.array([[False], [True], [False]]),
            'pre_frame': np.repeat(np.array([0., 1., 0.])[:, None, None], 45, axis=2),
            'post_frame': np.repeat(np.array([1., 2., 1.])[:, None, None], 45, axis=2),
            'normal_forces_w': np.zeros((3, 1, 6, 4, 3), dtype=np.float32),
        }
        for key in ('normal_peak_n', 'normal_integral_ns', 'positive_increment_n',
                    'positive_rate_fixed_window_n_per_s', 'positive_rate_one_step_n_per_s'):
            self.data[key] = np.zeros((3, 1, 4), dtype=np.float32)
        self.write()

    def write(self):
        path = self.root / 'shard_00000.npz'
        np.savez(path, **self.data)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        report = {
            'status': 'completed', 'num_envs': 1, 'steps': 3, 'reset_count': 1,
            'baseline_observer': False,
            'window': {'physics_dt': .005, 'policy_dt': .02, 'rise_interval_s': .01},
            'shards': [{'file': path.name, 'sha256': digest, 'steps': 3}]
        }
        (self.root / 'report.json').write_text(json.dumps(report))

    def test_terminal_observation_is_kept_without_cross_episode_comparison(self):
        result = audit_module.audit(self.root)
        self.assertEqual(result['valid_increment_rows'], 1)
        self.assertEqual(result['resets'], 1)
        self.assertEqual(result['post_to_next_pre_frame_max_error'], 0)

    def test_incomplete_run_is_not_a_dataset(self):
        (self.root / 'INCOMPLETE').touch()
        with self.assertRaisesRegex(ValueError, 'did not complete'):
            audit_module.audit(self.root)

    def test_corrupt_shard_is_rejected_before_loading(self):
        (self.root / 'shard_00000.npz').write_bytes(b'corrupted')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            audit_module.audit(self.root)

    def test_reset_difference_cannot_be_marked_valid(self):
        self.data['increment_valid'][2, 0] = True
        self.write()
        with self.assertRaisesRegex(ValueError, 'Reset padding'):
            audit_module.audit(self.root)

    def test_episode_age_must_restart(self):
        self.data['episode_step'][2, 0] = 2
        self.write()
        with self.assertRaisesRegex(ValueError, 'Episode IDs/ages'):
            audit_module.audit(self.root)

    def test_shifted_post_observation_is_rejected(self):
        self.data['post_frame'][0] = 99
        self.write()
        with self.assertRaisesRegex(ValueError, 'continuity'):
            audit_module.audit(self.root)
