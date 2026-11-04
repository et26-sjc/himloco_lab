"""Train independent stationary impact models on one identical cached stream.

Vectorized model axis accelerates comparisons; no parameters/Adam states are
shared. Five initializations use paired backbones and identical sampled data.
Final test data is neither loaded nor used by this script.
"""
import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch

from himloco_lab.impact.dataset import CausalDataset, file_hash
from himloco_lab.impact.metrics import fit_calibration
from himloco_lab.impact.models import ModelBank, outputs_and_losses


@torch.no_grad()
def predict(bank, data, ids, task, thresholds, batch_size=2048, details=True):
    score, point, probs, losses = [], [], [], []
    for start in range(0, len(ids), batch_size):
        batch_ids = ids[start:start + batch_size]
        x, y = data.batch(batch_ids, task)
        values = outputs_and_losses(bank(x, torch.as_tensor(batch_ids, device=data.device)), y, bank.entries, thresholds)
        losses.append(values[2].cpu().numpy() * len(batch_ids))
        if details:
            score.append(values[0].cpu().numpy())
            point.append(values[1].cpu().numpy())
            probs.append(values[3].cpu().numpy())
    result = {'losses': np.sum(losses, axis=0) / len(ids)}
    if details:
        result.update(scores=np.concatenate(score, axis=1), points=np.concatenate(point, axis=1),
                      probabilities=np.concatenate(probs, axis=1))
    return result


def run(args):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    spec = json.loads(args.spec.read_text())
    torch.manual_seed(20261103)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'INCOMPLETE').write_text('Training/calibration incomplete\n')
    (args.output / 'spec.json').write_text(json.dumps(spec, indent=2) + '\n')
    train = CausalDataset(args.train, args.device)
    val = CausalDataset(args.validation, args.device)
    if train.meta['collection_seed'] == val.meta['collection_seed']:
        raise ValueError('Training and validation geometry seeds overlap')
    if args.resume is not None:
        previous_report=json.loads((args.resume/'report.json').read_text())
        if (previous_report['status']!='completed' or
            previous_report['train_metadata_sha256']!=file_hash(args.train/'metadata.json') or
            previous_report['validation_metadata_sha256']!=file_hash(args.validation/'metadata.json')):
            raise ValueError('Resume requires the exact completed training/validation data')
    families = spec['training_family_ids']
    indices_np = train.indices(families)
    indices = torch.tensor(indices_np, device=args.device)
    rng = np.random.default_rng(777)
    count = spec['validation_samples_per_role']
    select_ids = rng.choice(val.indices(families, role=0), count, replace=False)
    calibration_ids = rng.choice(val.indices(families, role=1), count, replace=False)
    norm_ids = rng.choice(indices_np, count, replace=False)
    tx, _ = train.batch(norm_ids, 'forecast')
    normal_frames = tx.reshape(-1, 45)
    mean, std = normal_frames.mean(0), normal_frames.std(0).clamp_min(.01)
    thresholds = torch.tensor(spec['label']['thresholds_n'], device=args.device)
    all_runs = {}
    for task in spec['tasks']:
        out = args.output / task
        out.mkdir()
        bank = ModelBank(spec['models'], spec['training_seeds'], spec['hidden_sizes']).to(args.device)
        bank.mean.copy_(mean)
        bank.std.copy_(std)
        optimizer = torch.optim.Adam(bank.parameters(), lr=spec['learning_rate'])
        best = copy.deepcopy(bank.state_dict())
        best_loss = np.full(len(bank.entries), np.inf)
        best_steps = np.zeros(len(bank.entries), np.int64)
        history = []
        generator = torch.Generator(device=args.device).manual_seed(12345)
        initial_step = 0
        if args.resume is not None:
            previous_spec = json.loads((args.resume/'spec.json').read_text())
            for key in ('models','training_seeds','hidden_sizes','batch_size','label'):
                if previous_spec[key] != spec[key]:
                    raise ValueError(f'Incompatible resume: {key}')
            saved = torch.load(args.resume/task/'resume.pt',map_location=args.device,weights_only=False)
            bank.load_state_dict(saved['model'])
            optimizer.load_state_dict(saved['optimizer'])
            best, best_loss, best_steps = saved['best_model'], saved['best_loss'], saved['best_steps']
            initial_step = saved['step']
            if initial_step >= spec['optimizer_steps']:
                raise ValueError('Resume target steps must exceed the saved step')
            history = json.loads((args.resume/task/'learning_curve.json').read_text())
            generator.set_state(saved['generator_state'].cpu())
        torch.cuda.synchronize()
        start = time.perf_counter()
        unique = np.zeros(train.t * train.n, bool)
        if initial_step:
            # Earlier preparation checkpoints stored counts, not the bitset.
            # Replay the exact same CUDA randint calls to reconstruct exposure.
            reconstruction = torch.Generator(device=args.device).manual_seed(12345)
            for _ in range(initial_step):
                replay_ids = indices[torch.randint(len(indices), (spec['batch_size'],),
                                                  generator=reconstruction,device=args.device)]
                unique[replay_ids.cpu().numpy()] = True
            previous_report=json.loads((args.resume/'report.json').read_text())
            if int(unique.sum()) != previous_report['runs'][task]['unique_training_transitions_exposed']:
                raise ValueError('Resume data exposure reconstruction mismatch')
        for step in range(initial_step + 1, spec['optimizer_steps'] + 1):
            for milestone, rate in spec.get('learning_rate_milestones', []):
                if step == milestone:
                    for group in optimizer.param_groups:
                        group['lr'] = rate
            ids = indices[torch.randint(len(indices), (spec['batch_size'],), generator=generator, device=args.device)]
            x, y = train.batch(ids, task)
            optimizer.zero_grad(set_to_none=True)
            _, _, losses, _ = outputs_and_losses(bank(x, ids), y, bank.entries, thresholds)
            losses.sum().backward()
            optimizer.step()
            unique[ids.cpu().numpy()] = True
            if step % spec['validation_every'] == 0 or step == 1:
                selection = predict(bank, val, select_ids, task, thresholds, details=False)
                current = selection['losses']
                if not np.isfinite(current).all():
                    raise FloatingPointError('Non-finite validation loss')
                improved = current < best_loss
                best_loss[improved] = current[improved]
                best_steps[improved] = step
                mask = torch.as_tensor(improved, device=args.device)
                for key, value in bank.state_dict().items():
                    if key in ('mean', 'std', 'permutations'):
                        best[key].copy_(value)
                    else:
                        best[key][mask] = value[mask]
                history.append({'step': step, 'selection_losses': current.tolist(), 'train_losses': losses.detach().cpu().tolist(),
                                'elapsed_s': time.perf_counter() - start})
                (out / 'learning_curve.json').write_text(json.dumps(history, indent=2) + '\n')
                print(f'TRAIN_PROGRESS task={task} step={step} elapsed={time.perf_counter()-start:.1f}s best_last={int((best_steps==step).sum())}', flush=True)
        torch.save({'model': bank.state_dict(), 'optimizer': optimizer.state_dict(), 'step': step,
                    'generator_state': generator.get_state(), 'best_model': best,
                    'best_loss': best_loss, 'best_steps': best_steps}, out / 'resume.pt')
        np.save(out/'unique_exposed.npy',unique)
        bank.load_state_dict(best)
        torch.save({'model': best, 'entries': bank.entries, 'best_steps': best_steps.tolist()}, out / 'models.pt')
        calibration = predict(bank, val, calibration_ids, task, thresholds)
        target = np.asarray(val.cpu['severity']).reshape(-1, 4)[calibration_ids] > thresholds[1].item()
        calibrators = [fit_calibration(s, target) for s in calibration['scores']]
        if not all(p['success'] for p in calibrators):
            raise RuntimeError('Calibration optimizer did not converge')
        (out / 'calibration.json').write_text(json.dumps(calibrators, indent=2) + '\n')
        all_runs[task] = {'entries': bank.entries, 'best_steps': best_steps.tolist(),
                          'best_selection_losses': best_loss.tolist(), 'training_time_s': time.perf_counter() - start,
                          'unique_training_transitions_exposed': int(unique.sum()),
                          'unique_available_training_transitions': len(indices_np),
                          'sample_exposures_per_model': spec['optimizer_steps'] * spec['batch_size'],
                          'optimizer_steps_per_model': spec['optimizer_steps'],
                          'resumed_from_step': initial_step,
                          'models_sha256': file_hash(out / 'models.pt'),
                          'calibration_sha256': file_hash(out / 'calibration.json'),
                          'convergence_warning_at_budget_end': [bank.entries[i] for i, s in enumerate(best_steps)
                                                                if s >= spec['optimizer_steps'] - spec['validation_every']]}
        print('TASK_TRAINING_COMPLETE', task, flush=True)
        del bank, optimizer, best
        torch.cuda.empty_cache()
    report = {'status': 'completed', 'spec_sha256': file_hash(args.spec), 'runs': all_runs,
              'torch': torch.__version__, 'tf32': True, 'device': torch.cuda.get_device_name(),
              'paired_data_order': True, 'independent_data_replications': 1,
              'resumed_from': str(args.resume) if args.resume else None,
              'resume_report_sha256': file_hash(args.resume/'report.json') if args.resume else None,
              'train_metadata_sha256': file_hash(args.train / 'metadata.json'),
              'validation_metadata_sha256': file_hash(args.validation / 'metadata.json'),
              'source_sha256': {str(p): file_hash(p) for p in [Path(__file__), *Path('source/himloco_lab/himloco_lab/impact').glob('*.py')]}}
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    (args.output / 'INCOMPLETE').unlink()
    print('SUPERVISED_TRAINING_COMPLETE', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', type=Path, required=True)
    parser.add_argument('--validation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--spec', type=Path, default=Path('configs/impact/step1.json'))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--resume',type=Path)
    run(parser.parse_args())
