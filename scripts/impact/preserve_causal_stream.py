"""Preserve small reproducibility evidence; verified archive precedes bulk cleanup."""
import argparse
import json
from pathlib import Path
import tarfile

import numpy as np

from himloco_lab.impact.dataset import file_hash


ROOT=Path('datasets/go1_impact/causal_stream_v1')
ART=Path('artifacts/impact/causal_stream_v1')
LOG=Path('logs/impact_causal_stream_v1')


def compact():
    for cache in sorted(ROOT.glob('rep*/*/cache')):
        # prepare_events' metadata precedes adding confirmation/peak columns.
        # Preserve its byte-identical table and point the extended metadata at
        # the actual extended table instead of silently retaining a stale hash.
        metadata_path=cache/'metadata.json'
        metadata=json.loads(metadata_path.read_text())
        extended_sha=file_hash(cache/'events.npz')
        if metadata['events_sha256']!=extended_sha:
            base={k:v for k,v in dict(np.load(cache/'events.npz')).items()
                  if k not in ('confirm','native_peak_offset')}
            np.savez_compressed(cache/'event_alignment.npz',**base)
            assert file_hash(cache/'event_alignment.npz')==metadata['events_sha256']
            metadata['event_alignment_events_sha256']=metadata['events_sha256']
            metadata['event_alignment_filename']='event_alignment.npz'
            metadata['events_sha256']=extended_sha
            metadata['extension']='Original rows unchanged; causal confirmation and native peak offset added.'
            metadata_path.write_text(json.dumps(metadata,indent=2)+'\n')
        destination=cache/'evaluation_context.npz'
        if destination.exists():continue
        stream=dict(np.load(cache/'stream.npz'))
        np.savez_compressed(destination,age=stream['age'],done=stream['done'],valid=stream['valid'])
        native=np.load(cache/'native_audit.npz');ids=native['env_id']
        np.savez_compressed(cache/'audit_labels.npz',env_id=ids,
                            **{k:stream[k][:,ids] for k in ('state','pulse','event_id','valid')})


def archive(cleanup=False):
    assert (ART/'summary.json').exists(),'Formal final results required'
    compact()
    archive_dir=Path('artifacts/impact/archives/causal_stream_v1_20261108')
    archive_dir.mkdir(parents=True,exist_ok=True)
    files=[]
    for root in (ART,LOG):
        files.extend(p for p in root.rglob('*') if p.is_file() and 'development' not in p.parts and p.suffix!='.log')
    for cache in sorted(ROOT.glob('rep*/*/cache')):
        files.extend(p for p in cache.rglob('*') if p.is_file() and p.name not in ('pre.npy','stream.npz','predictions.npz'))
    for name in ('configs/impact/causal_stream_v1.json','docs/GO1_IMPACT_CAUSAL_STREAM_PLAN.md',
                 'docs/GO1_IMPACT_CAUSAL_STREAM_RESULTS.md','tests/test_causal_stream.py',
                 'source/himloco_lab/himloco_lab/impact/causal_stream.py',
                 'source/himloco_lab/himloco_lab/impact/observability.py',
                 'source/himloco_lab/himloco_lab/impact/dataset.py',
                 'source/himloco_lab/himloco_lab/impact/collection_cfg.py',
                 'source/himloco_lab/himloco_lab/impact/terrains.py',
                 'source/himloco_lab/himloco_lab/impact/labels.py',
                 'scripts/impact/collect.py','scripts/impact/train_force_study.py'):
        files.append(Path(name))
    files.extend(Path('scripts/impact').glob('*causal_stream*.py'))
    files=sorted(set(files));assert all(p.exists() for p in files)
    manifest={str(p):dict(bytes=p.stat().st_size,sha256=file_hash(p)) for p in files}
    target=archive_dir/'methods_models_events_streams.tar.gz'
    if not target.exists():
        with tarfile.open(target,'w:gz',compresslevel=3) as tar:
            for path in files:tar.add(path,arcname=str(path))
    import hashlib
    with tarfile.open(target,'r:gz') as tar:
        members=[m for m in tar.getmembers() if m.isfile()]
        assert {m.name for m in members}==set(manifest)
        for item in members:
            h=hashlib.sha256()
            with tar.extractfile(item) as stream:
                for block in iter(lambda:stream.read(8*1024**2),b''):h.update(block)
            assert h.hexdigest()==manifest[item.name]['sha256'],item.name
    report=dict(status='verified',archive=str(target),archive_bytes=target.stat().st_size,
                archive_sha256=file_hash(target),files=manifest)
    (archive_dir/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    if cleanup:
        removed=[]
        # Exact allowlist within this new experiment only. Never touches prior studies.
        for cache in sorted(ROOT.glob('*/**/cache')):
            for name in ('pre.npy','stream.npz','predictions.npz'):
                path=cache/name
                if path.exists():
                    removed.append(dict(path=str(path),bytes=path.stat().st_size,sha256=file_hash(path)))
                    path.unlink()
        report_cleanup=dict(status='completed',verified_archive_sha256=report['archive_sha256'],
                            removed_bytes=sum(r['bytes'] for r in removed),removed=removed)
        (archive_dir/'cleanup.json').write_text(json.dumps(report_cleanup,indent=2)+'\n')
    print('STREAM_ARCHIVE_VERIFIED',target,report['archive_bytes'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cleanup',action='store_true')
    a=p.parse_args();archive(a.cleanup)
