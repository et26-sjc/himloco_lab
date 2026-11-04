"""Lock alarm thresholds on validation, then evaluate independent test geometry.

Never changes fitted model weights or label definitions. Outputs all seed/method
results including failed operating points, rather than choosing a test winner.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from himloco_lab.impact.dataset import CausalDataset, file_hash
from himloco_lab.impact.metrics import (calibrate, event_metrics, frame_metrics,
                                       truth_event_ids, average_precision)
from himloco_lab.impact.models import ModelBank
from train import predict


def lock_operating_point(probability, valid, ids, starts, spec, task):
    candidates = np.unique(np.r_[np.quantile(probability[valid], [.5,.8,.9,.95,.98,.99,.995,.999,.9995,.9999]), 1.000001])
    rows = [event_metrics(probability, threshold, valid, ids, starts,
                          quiet_steps=spec['event']['alarm_refractory_steps'], task=task) for threshold in candidates]
    eligible = [row for row in rows if row['false_alarms_per_robot_minute'] <= spec['false_alarms_per_robot_minute']]
    best = max(eligible, key=lambda row: (row['event_recall'], -row['false_alarms_per_robot_minute']))
    return {'selected': best, 'candidates': rows,
            'rule': 'maximum validation recall satisfying FAR budget among fixed quantile grid; heldout geometry excluded'}


def bootstrap_comparisons(tile_aps, entries, groups, family_by_tile):
    results = {}
    lookup = {tuple(entry): i for i, entry in enumerate(entries)}
    seeds = sorted(set(seed for seed, _ in entries))
    rng = np.random.default_rng(20261103)
    for domain, families in groups.items():
        tile_mask = np.isin(family_by_tile, families)
        comparisons = {}
        for reference in ('bce', 'mse', 'huber', 'lognormal', 'independent'):
            difference = np.array([tile_aps[lookup[(seed,'ordinal')], tile_mask] -
                                   tile_aps[lookup[(seed,reference)], tile_mask] for seed in seeds])
            finite = np.isfinite(difference).all(0)
            difference = difference[:,finite]
            if difference.shape[1] == 0:
                continue
            samples = []
            for _ in range(1000):
                seed_ids = rng.integers(len(seeds),size=len(seeds))
                tile_ids = rng.integers(difference.shape[1],size=difference.shape[1])
                samples.append(float(difference[np.ix_(seed_ids,tile_ids)].mean()))
            comparisons[reference] = {'ordinal_minus_reference_macro_tile_AP': float(difference.mean()),
                'bootstrap_95pct': np.quantile(samples,[.025,.975]).tolist(),
                'per_initialization_differences': difference.mean(1).tolist(), 'tiles': int(finite.sum())}
        results[domain] = comparisons
    return {'results': results,
            'scope': 'crossed bootstrap over optimization initialization and test tiles, conditional on one test generation seed; not five data replications'}


def evaluate(args):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    spec = json.loads((args.training / 'spec.json').read_text())
    train_report = json.loads((args.training / 'report.json').read_text())
    if train_report['status'] != 'completed' or (args.training / 'INCOMPLETE').exists():
        raise ValueError('Incomplete training')
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'INCOMPLETE').write_text('Evaluation incomplete\n')
    val = CausalDataset(args.validation,args.device)
    test = CausalDataset(args.test,args.device)
    if test.meta['collection_seed'] in (spec['collection_seeds']['train'],val.meta['collection_seed']):
        raise ValueError('Test geometry seed overlap')
    thresholds = torch.tensor(spec['label']['thresholds_n'],device=args.device)
    seen = spec['training_family_ids']
    held = spec['heldout_family_ids']
    groups = {'all_seen': seen,'unseen_boxes':held}
    groups.update({name:[i] for i,name in enumerate(spec['terrain_families'])})
    val_env = np.flatnonzero(np.isin(val.family,seen) & (np.arange(val.n)%3==2))
    val_ids = val.indices(seen,role=2)
    val_valid = val.cpu['age'][:,val_env] >= spec['label']['ignore_episode_first_steps']
    val_truth, val_starts, _ = truth_event_ids(val.cpu['severity'][:,val_env],val_valid,
                                             merge_gap=spec['event']['merge_gap_steps'])
    test_ids = test.indices(range(len(spec['terrain_families'])))
    valid = test.cpu['age'] >= spec['label']['ignore_episode_first_steps']
    truth, starts, _ = truth_event_ids(test.cpu['severity'],valid,merge_gap=spec['event']['merge_gap_steps'])
    y = test.cpu['severity'].reshape(-1,4)[test_ids]
    precontact = test.cpu['precontact'].reshape(-1,4)[test_ids]
    touchdown = precontact & np.isfinite(test.cpu['onset_ms'].reshape(-1,4)[test_ids])
    family = test.family[test_ids % test.n]
    tile = test.tile[test_ids % test.n]
    tiles = np.unique(tile)
    family_by_tile = np.array([family[tile==t][0] for t in tiles])
    exposed = test.cpu['terrain_exposed'].reshape(-1)[test_ids]
    all_results = {}
    t0 = time.perf_counter()
    for task in spec['tasks']:
        bank = ModelBank(spec['models'],spec['training_seeds'],spec['hidden_sizes']).to(args.device)
        checkpoint = torch.load(args.training/task/'models.pt',map_location=args.device,weights_only=True)
        bank.load_state_dict(checkpoint['model'])
        calibrators = json.loads((args.training/task/'calibration.json').read_text())
        val_prediction = predict(bank,val,val_ids,task,thresholds)
        locked = []
        for i,entry in enumerate(bank.entries):
            full = np.zeros((*val_valid.shape,4),np.float32)
            full[val_valid] = calibrate(val_prediction['scores'][i],calibrators[i])
            locked.append(lock_operating_point(full,val_valid,val_truth,val_starts,spec,task))
        with (args.output/f'{task}_locked_thresholds.json').open('x') as stream:
            stream.write(json.dumps(locked,indent=2)+'\n')
        del val_prediction
        print('OPERATING_POINTS_LOCKED',task,flush=True)
        prediction = predict(bank,test,test_ids,task,thresholds)
        results = []
        tile_aps = np.full((len(bank.entries),len(tiles)),np.nan)
        for i,(seed,method) in enumerate(bank.entries):
            score = prediction['scores'][i]
            probability = calibrate(score,calibrators[i])
            point = prediction['points'][i]
            full = np.zeros((*valid.shape,4),np.float32)
            full[valid] = probability
            row = {'seed':seed,'method':method,'best_step':checkpoint['best_steps'][i],
                   'calibration':calibrators[i],'domains':{}}
            alarm_threshold = locked[i]['selected']['probability_threshold']
            for name,families in groups.items():
                sample_mask = np.isin(family,families)
                env_mask = np.isin(test.family,families)
                metrics = frame_metrics(y[sample_mask],score[sample_mask],probability[sample_mask],point[sample_mask])
                pc = precontact[sample_mask]
                metrics['precontact_foot_interval_AP'] = average_precision((y[sample_mask]>40)[pc],score[sample_mask][pc])
                metrics['precontact_prevalence'] = float((y[sample_mask]>40)[pc].mean())
                # Privileged selection is an evaluation stratum, never a model
                # input: can it distinguish hard vs soft upcoming touchdowns,
                # after conditioning away the simple touchdown-timing problem?
                td=touchdown[sample_mask]
                metrics['touchdown_conditioned_AP']=average_precision((y[sample_mask]>40)[td],score[sample_mask][td])
                metrics['touchdown_conditioned_prevalence']=float((y[sample_mask]>40)[td].mean())
                metrics['touchdown_foot_intervals']=int(td.sum())
                geometry_mask = sample_mask & exposed
                metrics['outside_spawn_platform_AP'] = average_precision(y[geometry_mask]>40,score[geometry_mask])
                metrics['events_at_validation_operating_point'] = event_metrics(
                    full[:,env_mask],alarm_threshold,valid[:,env_mask],truth[:,env_mask],starts[:,env_mask],
                    quiet_steps=spec['event']['alarm_refractory_steps'],task=task,
                    precontact=test.cpu['precontact'][:,env_mask],onset_ms=test.cpu['onset_ms'][:,env_mask])
                if method in ('independent','ordinal','lognormal'):
                    ps = prediction['probabilities'][i,sample_mask]
                    metrics['uncalibrated_exceedance_AP_by_threshold'] = [
                        average_precision(y[sample_mask]>tau,ps[...,j]) for j,tau in enumerate(spec['label']['thresholds_n'])]
                    metrics['severity_order_violation_fraction'] = float((ps[...,:-1]<ps[...,1:]).any(-1).mean())
                row['domains'][name] = metrics
            rng = np.random.default_rng(999)
            shuffled_labels = (y>40).reshape(-1).copy()
            rng.shuffle(shuffled_labels)
            row['randomized_pairing_AP_diagnostic'] = average_precision(shuffled_labels,score.reshape(-1))
            for j,tile_id in enumerate(tiles):
                mask=tile==tile_id
                ap=average_precision(y[mask]>40,score[mask])
                tile_aps[i,j]=np.nan if ap is None else ap
            results.append(row)
            print('TEST_EVALUATED',task,seed,method,
                  'seenAP',row['domains']['all_seen']['per_foot_AP'],
                  'unseenAP',row['domains']['unseen_boxes']['per_foot_AP'],flush=True)
        comparisons = bootstrap_comparisons(tile_aps,bank.entries,{'all_seen':seen,'unseen_boxes':held},family_by_tile)
        task_result = {'models':results,'paired_tile_comparisons':comparisons}
        (args.output/f'{task}_results.json').write_text(json.dumps(task_result,indent=2)+'\n')
        np.savez(args.output/f'{task}_tile_AP.npz',AP=tile_aps,tiles=tiles,families=family_by_tile)
        all_results[task]={'file':f'{task}_results.json','sha256':file_hash(args.output/f'{task}_results.json')}
        del prediction,bank
        torch.cuda.empty_cache()
    report={'status':'completed','tasks':all_results,'elapsed_s':time.perf_counter()-t0,
            'training_report_sha256':file_hash(args.training/'report.json'),
            'test_metadata_sha256':file_hash(args.test/'metadata.json'),
            'test_collection_seed':test.meta['collection_seed'],
            'limitations':['one data generation per split','simulation-only rapid-loading proxy',
                           'no replay yet','no controller effectiveness claim','1/min is a comparison operating point; test FAR can shift'],
            'source_sha256':{str(p):file_hash(p) for p in [Path(__file__), *Path('source/himloco_lab/himloco_lab/impact').glob('*.py')]}}
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    (args.output/'INCOMPLETE').unlink()
    print('TEST_EVALUATION_COMPLETE',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training',type=Path,required=True)
    p.add_argument('--validation',type=Path,required=True)
    p.add_argument('--test',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda:0')
    evaluate(p.parse_args())
