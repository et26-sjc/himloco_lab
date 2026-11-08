# Causal continuous Go1 impact monitoring — preservation index

The formal study is on `feat/go1-impact-causal-stream`. See the [full result and its limits](../../../../docs/GO1_IMPACT_CAUSAL_STREAM_RESULTS.md). Original locomotion and previous research files were left intact.

`methods_models_events_streams.tar.gz` is a **local, Git-ignored 5.2323 GiB bundle** containing 1023 verified files: all formal impact weights, protocol/source snapshot, reports, original/extended event tables, native-force audits, complete-input64-env fixtures, fixture outputs, compact evaluation context, and sparse alarm records. The archive's source snapshot matches the pre-test lock; the added test-label FROC/delay analyses are explicitly post-hoc. `manifest.json` records all SHA256 checksums and uncompressed sizes. Every tar member was read back and verified before cleanup.

`cleanup.json` records **21.6435 GiB** of removed full pre-observation, dense-label and full-prediction caches from this experiment only. `deduplication.json` records the separately removed unpacked copies of sparse alarm records; their verified copy remains inside the bundle. This avoids keeping duplicate multi-gigabyte records. The development run is excluded from formal evidence.

Locally retained outside the bundle:

- `logs/impact_causal_stream_v1/rep{1..5}/`: all final MLP/PD weights and frozen predecessor heads, about15 MiB total including logs.
- `datasets/go1_impact/causal_stream_v1/rep*/{train,validation,test,high_speed}/cache/`: small complete-input fixtures, native force audits, original/extended event tables and compact context. Bulk caches were removed.
- `artifacts/impact/causal_stream_v1/`: readable summaries, per-run/per-foot/per-tile statistics, model/working-point lock, verification report, diagnostic analyses and PNG/SVG figures. The large `.npz` alarm records are archived once.

To re-run the saved-record verification, restore the formal alarm records from the repository root, then invoke the original verifier:

```bash
tar -xzf artifacts/impact/archives/causal_stream_v1_20261108/methods_models_events_streams.tar.gz \
  --wildcards 'artifacts/impact/causal_stream_v1/rep*/test/*.npz' \
  'artifacts/impact/causal_stream_v1/rep*/high_speed/*.npz'
conda run -n himloco_lab python scripts/impact/verify_causal_stream.py
```

The original verification re-inferred all15 validation/test fixtures with maximum raw-output error0 and recomputed1360 final operating-point rows with metric error0. Full retraining requires regenerating the deleted simulation streams with the locked seeds; no previous220 GiB dataset is needed. Repeating the original orchestration in-place after cleanup requires a fresh output location or deliberate regeneration of this stage's caches, rather than pretending the deleted arrays remain available.
