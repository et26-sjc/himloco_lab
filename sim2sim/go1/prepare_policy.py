"""Preparation only: export/verify a checkpoint and bind a portable JIT manifest.

This command uses the existing HIM exporter. Normal run/evaluate commands do not
import the project, Isaac Lab, or the training checkpoint.
"""
import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from sim2sim.go1.policy_runner import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--policy', help='Verify an existing JIT instead of re-exporting')
    parser.add_argument('--output', help='New export directory when --policy is omitted')
    parser.add_argument('--profile', choices=['lab', 'legacy'], default='lab')
    args = parser.parse_args()
    if not args.policy and not args.output:
        parser.error('Supply --policy or --output')
    from himloco_lab.utils.checkpoint import load_inference_model
    from himloco_lab.utils.export_policy import export_himloco_policy_as_jit
    torch.set_num_threads(1)
    model, _ = load_inference_model(args.checkpoint)
    if args.policy:
        path = Path(args.policy)
    else:
        export_himloco_policy_as_jit(model, args.output)
        path = Path(args.output)/'policy.pt'
    jit = torch.jit.load(str(path), map_location='cpu').eval()
    generator = torch.Generator().manual_seed(17)
    errors = []
    with torch.inference_mode():
        for batch in (1, 7, 64):
            inputs = torch.randn(batch, 270, generator=generator)
            reference = model.act_inference(inputs)
            actual = jit(inputs)
            torch.testing.assert_close(actual, reference, atol=2e-5, rtol=2e-5)
            errors.append(float((actual-reference).abs().max()))
    record = dict(policy_sha256=sha256(path), checkpoint=str(Path(args.checkpoint).resolve()),
                  checkpoint_sha256=sha256(args.checkpoint), profile=args.profile,
                  input_shape=['batch', 270], output_shape=['batch', 12],
                  preparation='Strict checkpoint load and matching JIT inference, optimizer not restored',
                  pytorch_jit_max_abs_error=max(errors))
    path.with_suffix('.manifest.json').write_text(json.dumps(record, indent=2)+'\n')
    print('POLICY_MANIFEST_OK', json.dumps(record))


if __name__ == '__main__':
    main()
