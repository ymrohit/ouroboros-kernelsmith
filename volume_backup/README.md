# volume_backup — local mirror of the Modal volume `ouroboros-outputs`

- `datasets/verified_kernels.jsonl` — 843 verified kernels (earlier snapshot, original 31-op suite)
- `datasets_latest/verified_kernels_snapshot_1227.jsonl` — 1227 verified kernels (expanded-grammar snapshot)
- `reports/` — raw JSON verdicts from the recorded runs

NAMING NOTE (V2 fix): these files previously carried a `_2068` suffix that did not match
their line counts (the 2068 figure was the two snapshots summed, 843+1227≈2070, minus dedup).
Renamed to the actual entry counts; nothing else changed. The live corpus is rebuilt
deterministically by `sft_train.py --corpus-only` and the authoritative copy lives in the
Modal volume / the private HF dataset repo.
