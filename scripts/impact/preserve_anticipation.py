"""Verify a compact study archive before removing owned redundant staging."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
from zoneinfo import ZoneInfo

from himloco_lab.impact.dataset import file_hash
from collect_anticipation import ART,LOG,ROOT,SPEC
from evaluate_anticipation import dump


def preserve(cleanup=False):
    assert (ART/'summary.json').exists()
    assert Path('docs/GO1_IMPACT_SHARED_ANTICIPATION_RESULTS.md').exists()
    lock=json.loads((ART/'model_lock.json').read_text())
    for path,sha in {**lock['models'],**lock['operating_points']}.items():
        assert file_hash(path)==sha
    for rep in range(1,6):
        for role in ('validation','test','high_speed'):
            check=json.loads((ART/f'rep{rep}/{role}/verification.json').read_text())
            assert check['status']=='passed' and check['max_reinference_error']<1e-4
            if role!='validation':
                assert check['operating_rows_recounted']==552 and check['max_count_error']==0
    dest=Path('artifacts/impact/archives/shared_anticipation_v1_20261109')
    dest.mkdir(parents=True,exist_ok=True)
    bulk={'pre.npy','post.npy','next48.npy','current_targets.npy','prediction_staging.npy',
          'current_force_staging.npy','prediction_staging.json'}
    files=[]
    for root in (ART,LOG):
        files.extend(p for p in root.rglob('*') if p.is_file())
    for cache in ROOT.glob('*/**/cache'):
        files.extend(p for p in cache.rglob('*') if p.is_file() and p.name not in bulk)
    spec=json.loads(SPEC.read_text())
    files += [SPEC,Path(spec['checkpoint']),Path('go1_lab_deploy/models/policy.pt'),
              Path('scripts/impact/collect.py'),Path('scripts/impact/train_force_study.py'),
              Path('source/himloco_lab/himloco_lab/rsl_rl/modules/him_estimator.py')]
    files += list(Path('scripts/impact').glob('*anticipation*.py'))
    files += list(Path('source/himloco_lab/himloco_lab/impact').glob('*.py'))
    files += list(Path('docs').glob('GO1_IMPACT_*PLAN.md'))
    files += [Path('docs/GO1_IMPACT_SHARED_ANTICIPATION_RESULTS.md')]
    files += list(Path('tests').glob('test_impact*.py'))+[Path('tests/test_causal_stream.py')]
    files=sorted(set(files));assert all(p.is_file() and not p.is_symlink() for p in files)
    manifest={str(p):dict(bytes=p.stat().st_size,sha256=file_hash(p)) for p in files}
    target=dest/'methods_models_future_events.tar.gz'
    temporary=target.with_suffix('.tmp')
    if target.exists() and (dest/'manifest.json').exists():
        previous=json.loads((dest/'manifest.json').read_text())['files']
        assert all(previous.get(path)==record for path,record in manifest.items()),'Evidence changed after archive'
        missing=set(previous)-set(manifest)
        assert all(Path(path).name=='resume.pt' for path in missing)
        manifest=previous
    if not target.exists():
        with tarfile.open(temporary,'w:gz',compresslevel=1) as archive:
            for path in files:archive.add(path,arcname=str(path),recursive=False)
        temporary.rename(target)
    with tarfile.open(target,'r:gz') as archive:
        members=archive.getmembers()
        assert all(m.isfile() for m in members)
        assert {m.name for m in members}==set(manifest)
        for item in members:
            digest=hashlib.sha256()
            with archive.extractfile(item) as stream:
                for block in iter(lambda:stream.read(8*1024**2),b''):digest.update(block)
            assert digest.hexdigest()==manifest[item.name]['sha256'],item.name
    report=dict(status='verified',timestamp=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        archive=str(target),archive_bytes=target.stat().st_size,archive_sha256=file_hash(target),
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        files=manifest,scope='Local snapshot, not external backup. Full training requires recollection with recorded seeds/config; GPU PhysX bitwise determinism is not guaranteed. Complete fixtures allow independent inference and alarm recount.')
    dump(dest/'manifest.json',report)
    cleanup_done=(dest/'cleanup.json').exists() and json.loads((dest/'cleanup.json').read_text())['status']=='completed'
    if cleanup and not cleanup_done:
        removed=[]
        for cache in ROOT.glob('*/**/cache'):
            assert cache.resolve().is_relative_to(ROOT.resolve())
            for name in bulk:
                path=cache/name
                if path.is_file():
                    removed.append(dict(path=str(path),bytes=path.stat().st_size,sha256=file_hash(path)))
        # Optimizer/RNG states remain inside the verified archive; final models stay local.
        for path in LOG.rglob('resume.pt'):
            assert str(path) in manifest and file_hash(path)==manifest[str(path)]['sha256']
            removed.append(dict(path=str(path),bytes=path.stat().st_size,sha256=file_hash(path)))
        dump(dest/'cleanup.json',dict(status='planned_after_archive_readback',
            archive_sha256=report['archive_sha256'],removed_bytes=sum(r['bytes'] for r in removed),files=removed))
        for record in removed:Path(record['path']).unlink()
        cleanup_record=json.loads((dest/'cleanup.json').read_text());cleanup_record['status']='completed'
        dump(dest/'cleanup.json',cleanup_record)
    (dest/'README.md').write_text(
        '# Go1共享HIM未来冲击风险：保存索引\n\n'
        '本地压缩包已经逐成员SHA256回读校验。保留正式与开发权重、Adam/RNG、原baseline权重、完整64环境输入/输出片段、21环境原生接触审计、全部事件标签、锁定报警记录、报告和方法代码。\n\n'
        '模型仍可直接从`logs/impact_shared_anticipation_v1/rep*/models.pt`读取。全量输入大数组已清理；全量重新训练需依照固定seed和配置重新采集，GPU PhysX不保证逐位一致。现有权重和完整复核片段保留。此快照不是外部备份。\n\n'
        '主结果见[实验报告](../../../../docs/GO1_IMPACT_SHARED_ANTICIPATION_RESULTS.md)，详细文件和哈希见[manifest.json](manifest.json)。\n')
    print('ANTICIPATION_ARCHIVE_VERIFIED',target,report['archive_bytes'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--cleanup',action='store_true')
    preserve(parser.parse_args().cleanup)
