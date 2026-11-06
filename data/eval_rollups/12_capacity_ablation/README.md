# Capacity-Cliff Ablation (S3a setting)

Measures latent-attack capacity cliff under multi-bitrate-EOT recipe
(`qwen2_audio`, eps=1.0, 1000 steps) across target word count (Exp-A)
and carrier duration (Exp-B).

- **Spec:** `codec_attack/docs/superpowers/specs/2026-04-24-capacity-cliff-s3a-design.md`
- **Plan:** `codec_attack/docs/superpowers/plans/2026-04-24-capacity-cliff-s3a.md`
- **Status:** scaffolded — run `scripts/run_all.sh` to execute.

## Quickstart

```bash
cd ${REPO_ROOT}
bash 0_all_combined/results_ablation_capacity/scripts/run_all.sh
```

Expected wall-clock: ~2 h on one GPU. Resumable — each cell skips if already attacked.

## Artifacts (populated after `run_all.sh` finishes)

- `rollup_expA.json`, `rollup_expB.json` — aggregated per-cell success/WER/SNR
- `table_capacity_v2.tex` — NeurIPS-ready capacity table
- `figure_capacity.pdf` — two-panel cliff figure

## Sweep axes

| | Exp-A | Exp-B |
|--|--|--|
| Axis | target word count | carrier duration |
| Points | {2, 5, 8, 15, 20, 32, 40, 57, 82, 131} | {5, 10, 15, 20, 30, 60} s |
| Fixed | carrier dur = min(25, src) | 3 anchor lengths {5, 15, 40} words |
| Carriers | 3 Suno clips (jazz, classic, calm) | same |
| Cells | 30 | 54 |
