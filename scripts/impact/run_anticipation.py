"""Execute bounded sharing/anticipation study; lock all models before tests."""
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
from zoneinfo import ZoneInfo

import numpy as np

from himloco_lab.impact.dataset import file_hash
from collect_anticipation import collect,ROOT,ART,LOG,SPEC
from evaluate_anticipation import dump


def run(args,log):
    log=Path(log);log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('w') as stream:
        subprocess.run([sys.executable,*args],stdout=stream,stderr=subprocess.STDOUT,check=True)


def development_decision():
    dest=ART/'development/decision.json'
    if dest.exists():return
    report=json.loads((LOG/'development/training.json').read_text())
    assert report['updates']>=60000
    latency=json.loads((ART/'development/latency.json').read_text())
    spec=json.loads(SPEC.read_text())
    assert latency['maximum_p99_ms']<=spec['inference_allowance_ms']
    curve=json.loads((LOG/'development/curve.json').read_text());final=curve[-1]['models']
    selected={};values={}
    for loss in ('bce','brier','mse'):
        candidates={}
        for weight in (.1,1.,10.):
            chosen=[m['impact_loss'] for m in final if m['entry']['group']=='joint' and
                    m['entry']['loss']==loss and m['entry']['lambda']==weight]
            candidates[weight]=float(np.mean(chosen))
        selected[loss]=min(candidates,key=candidates.get);values[loss]=candidates
    previous=next(r for r in curve if r['step']==50000)
    relative=[(a['impact_loss']-b['impact_loss'])/max(a['impact_loss'],1e-12)
              for a,b in zip(previous['models'],final)]
    dump(dest,dict(status='locked_before_formal_training_and_tests',
        timestamp=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),selected_joint_lambda=selected,
        selection='Minimum own validation impact loss averaged across two initializations; validation seen-family role0 only.',
        validation_losses=values,formal_updates=60000,formal_batch=2048,
        initial_development_updates=15000,extended_development_updates=60000,
        reason_for_extension='Losses still improving at15000; continued same optimizer/RNG to60000 before formal tests.',
        last10000_relative_improvement_min_median_max=np.quantile(relative,[0,.5,1]).tolist(),
        all_lambda_sensitivity_models_retained=True,formal_models_per_replication=46,
        current_diagnostic='One replication matched completed20ms loading-rate diagnostic, not an independent five-seed primary.',
        lead_scope='20ms force-threshold research endpoint only. Geometry and actuator/N0 physical utility unverified.',
        host_inference_allowance_ms=spec['inference_allowance_ms'],
        development_maximum_batch1_p99_ms=latency['maximum_p99_ms'],
        development_training_sha256=file_hash(LOG/'development/training.json'),protocol_sha256=file_hash(SPEC)))


def cleanup_bulk(rep,roles):
    for role in roles:
        cache=ROOT/f'rep{rep}/{role}/cache'
        manifest=ART/f'rep{rep}/{role}/bulk_cleanup.json'
        if manifest.exists():
            for record in json.loads(manifest.read_text())['files']:
                path=Path(record['file'])
                if path.exists():
                    assert file_hash(path)==record['sha256'];path.unlink()
            continue
        assert cache.resolve().is_relative_to(ROOT.resolve())
        records=[]
        for name in ('pre.npy','post.npy','next48.npy','current_targets.npy'):
            path=cache/name
            if path.exists():
                records.append(dict(file=str(path),bytes=path.stat().st_size,sha256=file_hash(path)))
        dump(manifest,dict(status='verified_stage_bulk_removed',files=records,
            retained='Original labels, complete fixtures, native audits, provenance, models and locked sparse alarms.'))
        for record in records:Path(record['file']).unlink()


def main():
    ART.mkdir(parents=True,exist_ok=True);LOG.mkdir(parents=True,exist_ok=True)
    development_decision()
    for rep in range(1,6):
        for role in ('train','validation'):collect(rep,role)
        run(['scripts/impact/train_anticipation.py','--rep',str(rep)],LOG/f'rep{rep}_training.log')
        run(['scripts/impact/benchmark_anticipation.py','--rep',str(rep)],LOG/f'rep{rep}_benchmark.log')
        latency=json.loads((ART/f'rep{rep}/latency.json').read_text())
        assert latency['maximum_p99_ms']<=json.loads(SPEC.read_text())['inference_allowance_ms']
        if rep==1:
            run(['scripts/impact/train_anticipation.py','--rep','1','--current-diagnostic'],LOG/'rep1_current_diagnostic.log')
        run(['scripts/impact/evaluate_anticipation.py','--rep',str(rep),'--calibrate'],LOG/f'rep{rep}_calibration.log')
        run(['scripts/impact/verify_anticipation.py','--rep',str(rep),'--roles','validation'],LOG/f'rep{rep}_validation_verification.log')
        cleanup_bulk(rep,('train','validation'))
        print('ANTICIPATION_REP_LOCK_READY',rep,flush=True)
    sources=[SPEC,Path('docs/GO1_IMPACT_SHARED_REPRESENTATION_PLAN.md'),Path('docs/GO1_IMPACT_ANTICIPATION_PLAN.md'),
             Path('source/himloco_lab/himloco_lab/impact/anticipation.py'),
             Path('source/himloco_lab/himloco_lab/impact/observability.py'),
             Path('source/himloco_lab/himloco_lab/impact/causal_stream.py'),Path('scripts/impact/collect.py')]
    sources += [Path('scripts/impact/train_force_study.py'),Path('source/himloco_lab/himloco_lab/rsl_rl/modules/him_estimator.py')]
    sources+=list(Path('scripts/impact').glob('*anticipation*.py'))
    lock=dict(protocol_sha256=file_hash(SPEC),source_sha256={str(p):file_hash(p) for p in sources},
              decision_sha256=file_hash(ART/'development/decision.json'),models={},operating_points={})
    for rep in range(1,6):
        for name in ('models','phase'):
            path=LOG/f'rep{rep}/{name}.pt';lock['models'][str(path)]=file_hash(path)
        path=ART/f'rep{rep}/operating_points.json';lock['operating_points'][str(path)]=file_hash(path)
    output=ART/'model_lock.json'
    if output.exists():assert json.loads(output.read_text())==lock
    else:dump(output,lock)
    print('ANTICIPATION_ALL_MODELS_AND_POINTS_LOCKED',flush=True)
    for rep in range(1,6):
        if not (ART/f'rep{rep}/results.json').exists():
            for role in ('test','high_speed'):collect(rep,role)
            run(['scripts/impact/evaluate_anticipation.py','--rep',str(rep)],LOG/f'rep{rep}_evaluation.log')
            run(['scripts/impact/verify_anticipation.py','--rep',str(rep)],LOG/f'rep{rep}_verification.log')
        cleanup_bulk(rep,('test','high_speed'))
        print('ANTICIPATION_REP_EVALUATED',rep,flush=True)
    run(['scripts/impact/evaluate_anticipation.py','--summarize'],LOG/'summary.log')
    print('ANTICIPATION_COMPLETE',flush=True)


if __name__=='__main__':main()
