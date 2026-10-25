"""Disk JIT + newest-first history, independent of HIM Python classes and Isaac."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class HistoryBuffer:
    def __init__(self):
        self.frames = np.zeros((6, 45), dtype=np.float32)

    def reset(self):
        self.frames.fill(0)

    def push(self, frame):
        frame = np.asarray(frame, dtype=np.float32)
        if frame.shape != (45,) or not np.isfinite(frame).all():
            raise ValueError('Expected finite 45D current observation')
        self.frames[1:] = self.frames[:-1].copy()
        self.frames[0] = frame
        return self.frames.reshape(1, 270).copy()


class PolicyRunner:
    def __init__(self, path, profile='lab', checkpoint=None):
        self.path = Path(path).resolve()
        self.model = torch.jit.load(str(self.path), map_location='cpu').eval()
        manifest_path = self.path.with_suffix('.manifest.json')
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        if manifest and (manifest['policy_sha256'] != sha256(self.path) or manifest['profile'] != profile):
            raise ValueError('JIT hash or observation profile does not match its manifest')
        if checkpoint and manifest and sha256(checkpoint) != manifest['checkpoint_sha256']:
            raise ValueError('Checkpoint does not match the JIT manifest')
        self.metadata = dict(policy=str(self.path), policy_sha256=sha256(self.path), profile=profile,
                             checkpoint_sha256=sha256(checkpoint) if checkpoint else manifest.get('checkpoint_sha256'),
                             checkpoint=str(Path(checkpoint).resolve()) if checkpoint else manifest.get('checkpoint'))
        if self.metadata['checkpoint_sha256'] is None:
            raise ValueError('Supply a policy.manifest.json or --checkpoint for checkpoint provenance')
        self.history = HistoryBuffer()
        self.infer(np.zeros((1, 270), dtype=np.float32))

    def infer(self, history):
        if history.shape != (1, 270) or not np.isfinite(history).all():
            raise ValueError('Expected finite (1,270) history')
        with torch.inference_mode():
            output = self.model(torch.from_numpy(history)).cpu().numpy()
        if output.shape != (1, 12) or not np.isfinite(output).all():
            raise ValueError('Expected finite (1,12) action')
        return output[0].copy()

    def reset(self):
        self.history.reset()

    def act(self, frame):
        return self.infer(self.history.push(frame))
