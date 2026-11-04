"""CPU tests for causal alignment and physical-time candidate targets."""
import json
from pathlib import Path
import unittest

import torch

from himloco_lab.impact.labels import ContactWindow, completed_interval_frame, contact_window_targets


class ImpactPreparationTests(unittest.TestCase):
    def test_full_interval_includes_first_substep_and_prior_baseline(self):
        window = ContactWindow()
        self.assertEqual((window.decimation, window.rise_steps, window.history_length), (4, 2, 6))
        self.assertEqual(window.offsets_s, (-0.005, 0.0, 0.005, 0.01, 0.015, 0.02))
        forces = torch.zeros(1, 6, 4, 3)
        # The largest force is at the substep the baseline's 3-frame history loses.
        forces[:, 2, :, 0] = 100
        targets = contact_window_targets(forces, window)
        torch.testing.assert_close(targets['normal_peak_n'], torch.full((1, 4), 100.0))
        torch.testing.assert_close(targets['positive_increment_n'], torch.full((1, 4), 100.0))
        torch.testing.assert_close(targets['normal_integral_ns'], torch.full((1, 4), 0.5))
        torch.testing.assert_close(targets['positive_rate_one_step_n_per_s'], torch.full((1, 4), 20000.0))

    def test_steady_support_is_not_positive_loading(self):
        forces = torch.zeros(2, 6, 4, 3)
        forces[..., 2] = 30
        result = contact_window_targets(forces, ContactWindow())
        torch.testing.assert_close(result['positive_increment_n'], torch.zeros(2, 4))
        torch.testing.assert_close(result['normal_integral_ns'], torch.full((2, 4), 0.6))

    def test_fixed_physical_window_survives_dt_change_for_linear_loading(self):
        for dt in (0.005, 0.0025):
            window = ContactWindow(dt)
            forces = torch.zeros(1, window.history_length, 4, 3)
            times = torch.tensor(window.offsets_s)
            forces[..., 0] = (100 + 1000 * times)[None, :, None]
            result = contact_window_targets(forces, window)
            torch.testing.assert_close(result['positive_increment_n'], torch.full((1, 4), 10.0))
            torch.testing.assert_close(result['positive_rate_fixed_window_n_per_s'], torch.full((1, 4), 1000.0))

    def test_horizontal_normal_contact_is_preserved(self):
        forces = torch.zeros(1, 6, 4, 3)
        forces[:, 3:, :, 0] = 80
        result = contact_window_targets(forces, ContactWindow())
        torch.testing.assert_close(result['normal_peak_n'], torch.full((1, 4), 80.0))

    def test_terminal_frame_is_not_the_reset_frame(self):
        returned = torch.full((3, 270), 2.0)
        terminal = torch.full((3, 45), 9.0)
        ids = torch.tensor([1])
        completed = completed_interval_frame(returned, ids, terminal)
        torch.testing.assert_close(completed[1], terminal[1])
        torch.testing.assert_close(completed[0], returned[0, :45])
        self.assertTrue(torch.equal(returned, torch.full_like(returned, 2.0)))
        with self.assertRaises(ValueError):
            completed_interval_frame(returned, ids, None)

    def test_invalid_time_and_nonfinite_supervision_fail(self):
        for kwargs in ({'physics_dt': 0}, {'physics_dt': float('nan')}, {'physics_dt': 0.003}):
            with self.assertRaises(ValueError):
                ContactWindow(**kwargs)
        with self.assertRaises(ValueError):
            contact_window_targets(torch.zeros(1, 3, 4, 3), ContactWindow())
        forces = torch.zeros(1, 6, 4, 3)
        forces[0, 0, 0, 0] = float('nan')
        with self.assertRaises(ValueError):
            contact_window_targets(forces, ContactWindow())

    def test_plan_and_stationary_spec_agree_without_claiming_replay(self):
        root = Path(__file__).resolve().parents[1]
        plan = json.loads((root / 'configs/impact/go1_impact_replay.json').read_text())
        self.assertEqual(plan['collection']['num_envs'], 4096)
        spec = json.loads((root / 'configs/impact/step1.json').read_text())
        self.assertEqual(plan['labels']['binary_threshold_n'], spec['label']['thresholds_n'][1])
        self.assertEqual(plan['labels']['ordinal_thresholds_n'], spec['label']['thresholds_n'])
        self.assertEqual(plan['continual_comparison']['status'], 'planned_not_implemented')


if __name__ == '__main__':
    unittest.main()
