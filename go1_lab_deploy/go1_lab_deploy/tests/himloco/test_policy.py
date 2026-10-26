import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np

from go1_lab_deploy.himloco.policy import JitPolicy, sha256_file
from fixtures import release_config

try:
    import torch
except ImportError:
    torch = None


class PolicyValidationTests(unittest.TestCase):
    def test_hash_rejection_precedes_model_deserialization(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "not_a_model.pt"
            path.write_bytes(b"not a torchscript model")
            self.assertEqual(sha256_file(path), hashlib.sha256(path.read_bytes()).hexdigest())
            with self.assertRaisesRegex(ValueError, "SHA-256 does not match"):
                JitPolicy(path, "0" * 64)
            with self.assertRaisesRegex(ValueError, "64-character"):
                JitPolicy(path, None)

    def test_input_validation_without_loading_torch(self):
        # Rejections happen before any forward call or device interaction.
        policy = object.__new__(JitPolicy)
        for value in (np.zeros(270, dtype=np.float32), np.zeros((1, 270), dtype=np.float64),
                      np.full((1, 270), np.inf, dtype=np.float32)):
            with self.assertRaises(ValueError):
                policy(value)


@unittest.skipIf(torch is None, "PyTorch unavailable: synthetic TorchScript integration not executed")
class TorchScriptIntegrationTests(unittest.TestCase):
    def test_actual_pd_lab_jit_matches_direct_cpu_inference(self):
        config = release_config()
        path = config["model"]["path"]
        policy = JitPolicy(path, config["model"]["sha256"])
        independent = torch.jit.load(path, map_location="cpu").eval()
        rng = np.random.default_rng(20261026)
        inputs = [np.zeros((1, 270), dtype=np.float32)]
        inputs += [rng.normal(0., scale, (1, 270)).astype(np.float32) for scale in (.1, 1., 5.)]
        for history in inputs:
            with torch.no_grad():
                expected = independent(torch.from_numpy(history.copy())).cpu().numpy()[0]
            actual = policy(history)
            self.assertEqual(actual.shape, (12,))
            self.assertTrue(np.isfinite(actual).all())
            np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-6)
            np.testing.assert_array_equal(actual, policy(history.copy()))

    def test_single_module_cpu_raw_output_and_shape_validation(self):
        class SyntheticPolicy(torch.nn.Module):
            def forward(self, history):
                return history[:, :12] + 200.0

        class BadShapePolicy(torch.nn.Module):
            def forward(self, history):
                return history[:, :11]

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "synthetic_policy.pt"
            module = torch.jit.trace(SyntheticPolicy(), torch.zeros(1, 270))
            module.save(str(path))
            policy = JitPolicy(path, sha256_file(path))
            result = policy(np.zeros((1, 270), dtype=np.float32))
            np.testing.assert_array_equal(result, np.full(12, 200, dtype=np.float32))
            self.assertEqual(result.dtype, np.float32)
            bad_path = Path(folder) / "bad.pt"
            torch.jit.trace(BadShapePolicy(), torch.zeros(1, 270)).save(str(bad_path))
            with self.assertRaisesRegex(ValueError, "shape"):
                JitPolicy(bad_path, sha256_file(bad_path))

    def test_nonfinite_model_output_rejected_in_warmup(self):
        class NonfinitePolicy(torch.nn.Module):
            def forward(self, history):
                return history[:, :12] / 0.0

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "nonfinite.pt"
            torch.jit.trace(NonfinitePolicy(), torch.zeros(1, 270), check_trace=False).save(str(path))
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                JitPolicy(path, sha256_file(path))


if __name__ == "__main__":
    unittest.main()
