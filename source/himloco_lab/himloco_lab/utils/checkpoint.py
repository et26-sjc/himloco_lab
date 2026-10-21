"""Strict, tensor-only loading for inference; never load optimizer state here."""
from pathlib import Path
import torch
from himloco_lab.rsl_rl.modules import HIMActorCritic


def load_inference_model(checkpoint, device='cpu'):
    data = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if 'model_state_dict' not in data:
        raise ValueError('Checkpoint has no model_state_dict')
    state = data['model_state_dict']
    model = HIMActorCritic(270, 238, 45, 12)
    expected = model.state_dict()
    missing = sorted(set(expected) - set(state))
    unexpected = sorted(set(state) - set(expected))
    mismatch = {k: {'expected': list(expected[k].shape), 'actual': list(state[k].shape)}
                for k in expected.keys() & state.keys() if expected[k].shape != state[k].shape}
    if missing or unexpected or mismatch:
        raise ValueError(f'Strict checkpoint mismatch: missing={missing}, unexpected={unexpected}, shapes={mismatch}')
    model.load_state_dict(state, strict=True)
    if not all(torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError('Checkpoint contains non-finite parameters')
    metadata = dict(path=str(Path(checkpoint).resolve()), keys=list(data), iteration=data.get('iter'),
                    missing_keys=missing, unexpected_keys=unexpected, shape_mismatches=mismatch,
                    parameters={k: {'shape': list(v.shape), 'dtype': str(v.dtype)} for k, v in state.items()},
                    normalization='No running normalization parameters; act_inference uses the input directly.')
    return model.to(device).eval(), metadata
