"""Export and validate an existing checkpoint on CPU, without starting Isaac Sim."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import torch
from himloco_lab.utils.checkpoint import load_inference_model
from himloco_lab.utils.export_policy import export_himloco_policy_as_jit, export_himloco_policy_as_onnx

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', required=True)
parser.add_argument('--output', required=True)
parser.add_argument('--observations', required=True, help='Tensor or tensor dict captured from real rollout.')
parser.add_argument('--observation_key', default='lab_observations')
args = parser.parse_args()
torch.set_num_threads(4)
root = Path(args.output); root.mkdir(parents=True, exist_ok=True)
model, audit = load_inference_model(args.checkpoint)
reloaded, _ = load_inference_model(args.checkpoint)
export_himloco_policy_as_jit(model, str(root))
export_himloco_policy_as_onnx(model, str(root))
recorded = torch.load(args.observations, map_location='cpu', weights_only=True)
if isinstance(recorded, dict): recorded = recorded[args.observation_key]
assert recorded.ndim == 2 and recorded.shape[1] == 270
assert torch.isfinite(recorded).all()
torch.manual_seed(19)
cases = {f'random_batch_{n}': torch.randn(n, 270) for n in (1, 7, 64)}
cases['real_rollout'] = recorded[:256].contiguous()
torch.save(cases, root/'validation_inputs.pt')
# Fresh isolated process proves the disk exports have no Python HIM/Lab dependency.
completed = subprocess.run([sys.executable, '-I', str(Path(__file__).with_name('standalone_inference.py')),
                           '--exported', str(root.resolve()), '--inputs', str((root/'validation_inputs.pt').resolve()),
                           '--output', str((root/'standalone_outputs.pt').resolve())], check=True, capture_output=True, text=True)
print(completed.stdout)
outputs = torch.load(root/'standalone_outputs.pt', weights_only=True)
report = dict(checkpoint=audit, observations_source=str(Path(args.observations).resolve()),
              observation_key=args.observation_key, standalone_process=True, backend='CPU',
              onnx_interfaces={'encoder': ['batch,270', 'batch,19'], 'actor': ['batch,64', 'batch,12']}, cases={})
with torch.inference_mode():
    for name, inputs in cases.items():
        reference = model.act_inference(inputs)
        comparisons = {'reloaded_pytorch': reloaded.act_inference(inputs), **outputs[name]}
        report['cases'][name] = {}
        for backend, actual in comparisons.items():
            torch.testing.assert_close(actual, reference, atol=3e-5, rtol=3e-5)
            error = (actual-reference).abs()
            report['cases'][name][backend] = dict(max_abs_error=error.max().item(), mean_abs_error=error.mean().item(),
                                                   shape=list(actual.shape))
(root/'validation.json').write_text(json.dumps(report, indent=2))
print('EXPORT_VALIDATION_OK', json.dumps(report['cases']))
