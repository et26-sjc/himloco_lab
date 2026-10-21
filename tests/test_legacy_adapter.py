import unittest
import torch
from himloco_lab.inference.legacy_go1 import (LegacyGo1ObservationAdapter, LEGACY_JOINT_NAMES,
                                            LEGACY_DEFAULT_POSITIONS)

LAB_NAMES = [f'{leg}_{joint}_joint' for joint in ('hip', 'thigh', 'calf') for leg in ('FL', 'FR', 'RL', 'RR')]

class LegacyAdapterTests(unittest.TestCase):
    def test_named_mapping_and_scales(self):
        adapter = LegacyGo1ObservationAdapter(LAB_NAMES, LAB_NAMES, LEGACY_DEFAULT_POSITIONS)
        frame = torch.zeros(2, 45); frame[:, :3] = torch.tensor([.7, .2, .3])
        frame[:, 8] = -1
        for i, name in enumerate(LAB_NAMES):
            frame[:, 9+i] = LEGACY_JOINT_NAMES.index(name) * .01
            frame[:, 21+i] = LEGACY_JOINT_NAMES.index(name) * .1
            frame[:, 33+i] = LEGACY_JOINT_NAMES.index(name)
        old = adapter.adapt_frame(frame)
        torch.testing.assert_close(old[0, :3], torch.tensor([1.4, .4, .075]))
        torch.testing.assert_close(old[0, 9:21], torch.arange(12) * .01)
        torch.testing.assert_close(old[0, 21:33], torch.arange(12) * .1)
        torch.testing.assert_close(old[0, 33:45], torch.arange(12).float())
        action = adapter.to_lab_action(torch.arange(12).float().unsqueeze(0))
        torch.testing.assert_close(action[0], torch.tensor([LEGACY_JOINT_NAMES.index(n) for n in LAB_NAMES]).float())

    def test_defaults_action_offsets_and_inverse(self):
        defaults = {name: value+.2 for name, value in LEGACY_DEFAULT_POSITIONS.items()}
        adapter = LegacyGo1ObservationAdapter(LAB_NAMES, list(reversed(LAB_NAMES)), defaults, lab_action_scale=.5)
        old = torch.randn(3, 12)
        lab = adapter.to_lab_action(old)
        frame = torch.zeros(3, 45); frame[:, 8] = -1; frame[:, 33:45] = lab
        converted = adapter.adapt_frame(frame)
        torch.testing.assert_close(converted[:, 33:45], old, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(converted[:, 9:21], torch.full((3, 12), .2), atol=1e-6, rtol=0)

    def test_history_order_padding_and_reset(self):
        adapter = LegacyGo1ObservationAdapter(LAB_NAMES, LAB_NAMES, LEGACY_DEFAULT_POSITIONS)
        frames = torch.zeros(2, 6, 45)
        frames[:, :3, 8] = -1
        frames[:, :3, 0] = torch.tensor([3., 2., 1.])
        out = adapter.adapt_history(frames.flatten(1), torch.tensor([0])).reshape(2, 6, 45)
        torch.testing.assert_close(out[0, :, 0], torch.tensor([6., 4., 2., 0., 0., 0.]))
        self.assertEqual(torch.count_nonzero(out[:, 3:]).item(), 0)
        self.assertEqual(torch.count_nonzero(out[0, 0, 33:]).item(), 0)
        torch.testing.assert_close(frames[:, :3, 0], torch.tensor([[3., 2., 1.], [3., 2., 1.]]))

if __name__ == '__main__':
    unittest.main()
