"""Causal indexing, independent optimization, and event accounting contracts."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from himloco_lab.impact.dataset import CausalDataset, smoothed_loading_rise
from himloco_lab.impact.metrics import average_precision, event_metrics, truth_event_ids
from himloco_lab.impact.models import ModelBank, outputs_and_losses


class ImpactLearningTests(unittest.TestCase):
    def test_filter_preserves_equal_impulse_pulse_when_dt_halves(self):
        for lag, decimation, amplitude in [(2, 4, 200.), (4, 8, 400.)]:
            before = np.zeros((1, 2 * lag, 4))
            now = np.zeros((1, decimation, 4))
            now[:, 0] = amplitude
            np.testing.assert_allclose(smoothed_loading_rise(before, now, lag), 100.)

    def test_steady_support_and_linear_loading(self):
        for lag, dt in [(2, .005), (4, .0025)]:
            time = np.arange(-2 * lag + 1, 2 * lag + 1) * dt
            force = (100. + 1000. * time)[None, :, None].repeat(4, axis=2)
            np.testing.assert_allclose(smoothed_loading_rise(force[:, :2*lag], force[:, 2*lag:], lag), 10.)
            np.testing.assert_allclose(smoothed_loading_rise(force[:, :2*lag]*0+30, force[:, 2*lag:]*0+30, lag), 0.)

    def test_causal_inputs_and_reset_padding(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            meta = {'status': 'completed', 'steps': 8, 'num_envs': 1,
                    'label': {'ignore_episode_first_steps': 0},
                    'terrain': {'family_id': [0], 'tile_id': [0]}}
            (root/'metadata.json').write_text(json.dumps(meta))
            frames = np.broadcast_to(np.arange(8, dtype=np.float32)[:, None, None], (8, 1, 45))
            for key in ('pre', 'post'):
                np.save(root/f'{key}.npy', frames + (100 if key == 'post' else 0))
            for key in ('severity','precontact','onset_ms'):
                np.save(root/f'{key}.npy', np.zeros((8,1,4),np.float32))
            for key in ('done','terminated','terrain_exposed'):
                np.save(root/f'{key}.npy', np.zeros((8,1),bool))
            np.save(root/'age.npy', np.array([0,1,2,3,0,1,2,3])[:,None])
            data = CausalDataset(root, 'cpu')
            d, _ = data.batch([6], 'detection')
            p, _ = data.batch([6], 'forecast')
            self.assertEqual(d.reshape(6,45)[:,0].tolist(), [106.,6.,5.,4.,0.,0.])
            self.assertEqual(p.reshape(6,45)[:,0].tolist(), [6.,5.,4.,0.,0.,0.])

    def test_model_axis_gradients_are_independent(self):
        torch.manual_seed(30)
        x, y, ids = torch.randn(7,270), torch.rand(7,4)*100, torch.arange(7)
        thresholds = torch.tensor([20.,40.,80.])
        bank = ModelBank(['bce','ordinal'], [1,2])
        single = ModelBank(['bce'], [1])
        outputs_and_losses(bank(x,ids), y, bank.entries, thresholds)[2].sum().backward()
        outputs_and_losses(single(x,ids), y, single.entries, thresholds)[2].sum().backward()
        for name, parameter in bank.named_parameters():
            torch.testing.assert_close(parameter.grad[0], dict(single.named_parameters())[name].grad[0])
        ps = outputs_and_losses(bank(x,ids), y, bank.entries, thresholds)[3]
        self.assertTrue(torch.all(ps[[1,3], ..., :-1] >= ps[[1,3], ..., 1:]))

    def test_tied_ranking_does_not_invent_information(self):
        self.assertEqual(average_precision([0,1,0,1], [0,0,0,0]), .5)
        self.assertEqual(average_precision([0,1,0,1], [0,1,0,1]), 1.)

    def test_event_matching_duplicates_quiet_period_and_no_future_tolerance(self):
        severity = np.zeros((20,1,4)); severity[5:8,0,0] = 60
        valid = np.ones((20,1),bool)
        ids, starts, total = truth_event_ids(severity, valid)
        self.assertEqual(total, 1)
        probability = np.zeros_like(severity)
        probability[[4,6,15],0,0] = .9
        result = event_metrics(probability,.8,valid,ids,starts,quiet_steps=0)
        self.assertEqual((result['alarms'],result['matched_events'],result['false_alarms']), (3,1,2))
        quiet = event_metrics(probability,.8,valid,ids,starts,quiet_steps=5)
        self.assertEqual((quiet['alarms'],quiet['matched_events']), (2,0))

    def test_truth_merge_never_bridges_invalid_reset(self):
        severity = np.zeros((8,1,4));severity[[2,4],0,0]=60
        valid=np.ones((8,1),bool)
        self.assertEqual(truth_event_ids(severity,valid)[2],1)
        valid[3]=False
        self.assertEqual(truth_event_ids(severity,valid)[2],2)


if __name__ == '__main__':
    unittest.main()
