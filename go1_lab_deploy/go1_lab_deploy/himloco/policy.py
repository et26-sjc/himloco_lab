"""One exported HIM TorchScript module; no parameters.pkl or external estimator."""

import hashlib
import io
import os
import re

import numpy as np

from .contract import ACTION_DIM, HISTORY_DIM


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class JitPolicy:
    """CPU inference with exact input/output validation and explicit file hash."""

    def __init__(self, model_path, expected_sha256):
        if not isinstance(expected_sha256, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", expected_sha256):
            raise ValueError("model.sha256 must be the expected 64-character SHA-256")
        self.model_path = os.path.abspath(os.fspath(model_path))
        with open(self.model_path, "rb") as stream:
            model_bytes = stream.read()
        self.sha256 = hashlib.sha256(model_bytes).hexdigest()
        if self.sha256 != expected_sha256.lower():
            raise ValueError("policy SHA-256 does not match model.sha256")
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("PyTorch is required to load policy.pt; install a compatible CPU runtime") from exc
        self._torch = torch
        torch.set_num_threads(1)
        # Load exactly the bytes whose hash was checked, even if a transfer
        # replaces the on-disk file concurrently with startup.
        self._module = torch.jit.load(io.BytesIO(model_bytes), map_location="cpu")
        self._module.eval()
        for _ in range(5):
            self(np.zeros((1, HISTORY_DIM), dtype=np.float32))

    def __call__(self, history):
        value = np.asarray(history)
        if value.shape != (1, HISTORY_DIM) or value.dtype != np.float32:
            raise ValueError("JIT input must have shape (1,270) and dtype float32")
        if not np.all(np.isfinite(value)):
            raise ValueError("JIT input contains nonfinite values")
        tensor = self._torch.from_numpy(np.ascontiguousarray(value.copy()))
        with self._torch.no_grad():
            result = self._module(tensor)
        if not isinstance(result, self._torch.Tensor):
            raise ValueError("HIM JIT must return a Tensor, not a tuple or dictionary")
        if tuple(result.shape) != (1, ACTION_DIM) or result.dtype != self._torch.float32:
            raise ValueError("HIM JIT output must have shape (1,12) and dtype float32")
        output = result.detach().cpu().numpy()[0].copy()
        if not np.all(np.isfinite(output)):
            raise ValueError("HIM JIT output contains nonfinite values")
        # The caller applies Lab's affine target transform/clip and SDK mapping.
        return output
