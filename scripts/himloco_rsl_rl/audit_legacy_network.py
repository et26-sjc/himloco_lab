"""Compare the original and migrated HIM source on identical CPU tensors."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import torch
from himloco_lab.utils.checkpoint import load_inference_model

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', required=True)
parser.add_argument('--baseline', default='/home/sjc/HIMLoco-Go1')
parser.add_argument('--output', default='artifacts/round2/network_audit.json')
args = parser.parse_args()
torch.set_num_threads(4)
torch.manual_seed(7)
model, report = load_inference_model(args.checkpoint)
sys.path.insert(0, str(Path(args.baseline) / 'rsl_rl'))
from rsl_rl.modules.him_actor_critic import HIMActorCritic as OriginalHIMActorCritic
original = OriginalHIMActorCritic(270, 238, 45, 12).eval()
original.load_state_dict(torch.load(args.checkpoint, map_location='cpu', weights_only=True)['model_state_dict'], strict=True)
inputs = torch.randn(64, 270)
with torch.inference_mode():
    action = model.act_inference(inputs)
    reference = original.act_inference(inputs)
    velocity, latent = model.estimator(inputs)
    torch.testing.assert_close(action, model.act_inference(inputs), rtol=0, atol=0)
    torch.testing.assert_close(action, reference, rtol=0, atol=0)
assert action.shape == (64, 12) and velocity.shape == (64, 3) and latent.shape == (64, 16)
assert all(torch.isfinite(x).all() for x in (action, velocity, latent))
report.update(strict_load=True, batch=64, max_abs_error=(action-reference).abs().max().item(),
              mean_abs_error=(action-reference).abs().mean().item(), deterministic=True,
              sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest())
Path(args.output).parent.mkdir(parents=True, exist_ok=True)
Path(args.output).write_text(json.dumps(report, indent=2))
print('NETWORK_AUDIT_OK', json.dumps({k:v for k,v in report.items() if k != 'parameters'}))
