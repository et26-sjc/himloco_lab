"""Only torch / ONNX Runtime, no HIM source, Runner, Critic, or Isaac Lab imports.

Can be copied with exported files to a machine without Isaac Lab.
"""
import argparse
import json
from pathlib import Path
import sys
import torch
import onnxruntime as ort
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--exported', required=True)
parser.add_argument('--inputs', required=True)
parser.add_argument('--output', required=True)
args = parser.parse_args()
torch.set_num_threads(4)
root = Path(args.exported)
model = torch.jit.load(str(root/'policy.pt'), map_location='cpu').eval()
options = ort.SessionOptions(); options.intra_op_num_threads = 4; options.inter_op_num_threads = 1
encoder = ort.InferenceSession(str(root/'encoder.onnx'), options, providers=['CPUExecutionProvider'])
actor = ort.InferenceSession(str(root/'policy.onnx'), options, providers=['CPUExecutionProvider'])
assert encoder.get_inputs()[0].shape[1] == 270
assert actor.get_inputs()[0].shape[1] == 64
result = {}
with torch.inference_mode():
    for name, inputs in torch.load(args.inputs, map_location='cpu', weights_only=True).items():
        jit = model(inputs)
        encoded = encoder.run(None, {'obs_history': inputs.numpy()})[0]
        assert encoded.shape == (len(inputs), 19)
        actor_input = np.concatenate((inputs.numpy()[:, :45], encoded), axis=-1)
        onnx = actor.run(None, {'obs': actor_input})[0]
        assert jit.shape == onnx.shape == (len(inputs), 12)
        assert torch.isfinite(jit).all() and np.isfinite(onnx).all()
        result[name] = dict(jit=jit, onnx=torch.from_numpy(onnx))
assert not any(n.startswith(('isaaclab', 'isaacsim', 'himloco_lab')) for n in sys.modules)
torch.save(result, args.output)
print('STANDALONE_INFERENCE_OK', json.dumps({'encoder_input': encoder.get_inputs()[0].shape,
                                          'actor_input': actor.get_inputs()[0].shape,
                                          'batches': {k:len(v['jit']) for k,v in result.items()}}))
