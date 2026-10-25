"""Shared command line without training or Isaac launcher dependencies."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .config import load_config
from .policy_runner import PolicyRunner


def parser(description):
    result = argparse.ArgumentParser(description=description)
    result.add_argument('--policy', required=True, help='Combined Estimator+Actor TorchScript JIT')
    result.add_argument('--checkpoint', help='Source checksum if the JIT has no provenance manifest')
    result.add_argument('--config', help='Optional YAML; explicitly supplied CLI values take precedence')
    result.add_argument('--model')
    result.add_argument('--profile', choices=['lab', 'legacy'], default=None)
    result.add_argument('--kp', type=float, default=None)
    result.add_argument('--kd', type=float, default=None)
    result.add_argument('--passive-damping', type=float, default=None,
                        help='Explicit physical joint damping override; 0 removes the extra Menagerie damping. Kd is unchanged.')
    result.add_argument('--duration', type=float, default=20.)
    result.add_argument('--velocity', type=float, nargs=3, default=None)
    result.add_argument('--output', default='artifacts/sim2sim/go1_run')
    return result


def setup(args):
    torch.set_num_threads(1)
    cfg = load_config(args.config, model=args.model, profile=args.profile, kp=args.kp, kd=args.kd,
                      passive_damping=args.passive_damping)
    policy = PolicyRunner(args.policy, cfg.profile, args.checkpoint)
    print('EFFECTIVE_CONFIG', json.dumps(cfg.to_dict()), flush=True)
    print('POLICY_PROVENANCE', json.dumps(policy.metadata), flush=True)
    return cfg, policy


def save_result(root, metrics, trace):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    (root/'results.json').write_text(json.dumps(metrics, indent=2, allow_nan=False)+'\n')
    np.savez_compressed(root/'trace.npz', **trace)
