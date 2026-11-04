"""Preserve methods, valid models, compact events and scores; remove bulk frames."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile

from himloco_lab.impact.dataset import file_hash


def main(clean=False):
    root=Path('artifacts/impact/observability_v1')
    assert (root/'summary.json').exists()
    archive=Path('artifacts/impact/archives/observability_v1_20261105');archive.mkdir(parents=True,exist_ok=True)
    source=[Path('configs/impact/observability_v1.json'),Path('docs/GO1_IMPACT_OBSERVABILITY_PLAN.md'),
            Path('docs/GO1_IMPACT_OBSERVABILITY_RESULTS.md'),Path('source/himloco_lab/himloco_lab/impact/observability.py'),
            Path('scripts/impact/collect.py'),Path('tests/test_impact_observability.py')]
    source+=list(Path('scripts/impact').glob('*observability*.py'))
    source+=list(Path('source/himloco_lab/himloco_lab/impact').glob('*.py'))
    source+=[Path('scripts/impact/train_force_study.py'),Path('configs/impact/go1_impact_replay.json')]
    source+=list(root.rglob('*.json'))+list(root.rglob('*.png'))+list(root.rglob('*.npz'))
    source+=list(root.rglob('*.txt'))
    for rep in range(1,6):
        source+=list((Path('logs/impact_observability_v1')/f'rep{rep}').rglob('*.pt'))
        source+=list((Path('logs/impact_observability_v1')/f'rep{rep}').rglob('*.json'))
    for path in Path('datasets/go1_impact/observability_v1').rglob('*'):
        if path.is_file() and path.name in ('events.npz','native_audit.npz','metadata.json'):source.append(path)
    source=sorted(set(source))
    manifest={'files':{str(p):{'sha256':file_hash(p),'bytes':p.stat().st_size} for p in source}}
    bundle=archive/'methods_models_events_scores.tar.gz'
    if not bundle.exists():
        with tarfile.open(bundle,'w:gz',compresslevel=3) as tar:
            for path in source:tar.add(path,arcname=str(path),recursive=False)
    with tarfile.open(bundle,'r:gz') as tar:
        for member in tar:
            stream=tar.extractfile(member)
            digest=hashlib.file_digest(stream,'sha256').hexdigest()
            assert manifest['files'][member.name]['sha256']==digest,member.name
    manifest.update(bundle_sha256=file_hash(bundle),bundle_bytes=bundle.stat().st_size,
                    readback_verified=True,files_count=len(source))
    (archive/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    removed=[]
    if clean:
        for path in Path('datasets/go1_impact/observability_v1').rglob('pre.npy'):
            meta=json.loads((path.parent/'metadata.json').read_text())
            assert file_hash(path)==meta['pre_sha256'],path
            removed.append({'path':str(path),'bytes':path.stat().st_size,'sha256':meta['pre_sha256']})
            path.unlink()
    removal={'removed':removed,'removed_bytes':sum(x['bytes'] for x in removed),
             'note':'Only this experiment generated full pre-frame arrays removed; event tables, predictions, audits, models and previous research kept.'}
    if removed:(archive/'DATA_REMOVAL.json').write_text(json.dumps(removal,indent=2)+'\n')
    (archive/'README.md').write_text('# Go1 observability 保存索引\n\n'
        '本地归档 `methods_models_events_scores.tar.gz` 保存方法、240个正式独立模型、事件表、各方法预测、小规模原生审计与报告。压缩包完整读回校验，文件哈希见 manifest.json。大文件留本地，代码和报告可提交分支。\n\n'
        '清理清单见 DATA_REMOVAL.json；只删除本轮生成的全量本体帧。保留事件标签和预测允许重算事件级指标，原生审计允许抽样复核目标与时间边界；重新训练或从完整观测重跑推理仍需重新采集，不能保证PhysX逐比特一致。\n')
    print('OBS_ARCHIVE',json.dumps({k:v for k,v in manifest.items() if k!='files'}),
          'removed_GiB',removal['removed_bytes']/2**30,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--clean',action='store_true')
    a=p.parse_args();main(a.clean)
