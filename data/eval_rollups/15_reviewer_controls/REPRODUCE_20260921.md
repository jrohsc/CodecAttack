# Reproduce the controlled S1 comparisons

Working directory:

```bash
cd ${REPO_ROOT}
```

## Existing N=1/N=2/N=3 ensemble ladder

```bash
n1a=$(sbatch --parsable --export=ALL,POOL_LABEL=n1_qwen2,SHARD_ID=0 sbatch_s1_pool_opuseot_eps10_shard.sh)
n1b=$(sbatch --parsable --export=ALL,POOL_LABEL=n1_qwen2,SHARD_ID=1 sbatch_s1_pool_opuseot_eps10_shard.sh)
sbatch --dependency=afterok:$n1a:$n1b --export=ALL,POOL_LABEL=n1_qwen2 sbatch_s1_pool_opuseot_eps10_eval.sh

n2a=$(sbatch --parsable --export=ALL,POOL_LABEL=n2_qwen2_omni,SHARD_ID=0 sbatch_s1_pool_opuseot_eps10_shard.sh)
n2b=$(sbatch --parsable --export=ALL,POOL_LABEL=n2_qwen2_omni,SHARD_ID=1 sbatch_s1_pool_opuseot_eps10_shard.sh)
sbatch --dependency=afterok:$n2a:$n2b --export=ALL,POOL_LABEL=n2_qwen2_omni sbatch_s1_pool_opuseot_eps10_eval.sh

n3a=$(sbatch --parsable --export=ALL,SHARD_ID=0 sbatch_s1_ens3_opuseot_eps10_shard.sh)
n3b=$(sbatch --parsable --export=ALL,SHARD_ID=1 sbatch_s1_ens3_opuseot_eps10_shard.sh)
sbatch --dependency=afterok:$n3a:$n3b sbatch_s1_ens3_opuseot_eps10_eval.sh
```

These commands reproduce the current descriptive ladder. They do not correct
the unseeded process-global Opus bitrate RNG documented in
`PROTOCOL_AUDIT_20260921.md`.

After all three evaluation jobs finish:

```bash
sbatch sbatch_s1_ensemble_report.sh
```

## Missing arm C: tuned unrestricted waveform

The frozen development choice is alpha/epsilon=0.02. The following runs only
the 40-example held-out split. It never reads held-out outputs during attack or
distortion calibration.

```bash
probe=()
for shard in 0 1 2 3; do
  probe+=("$(sbatch --parsable --export=ALL,PHASE=probe,SHARD_ID=$shard sbatch_s1_tuned_full40_shard.sh)")
done
probe_dep=$(IFS=:; echo "${probe[*]}")
cal=$(sbatch --parsable --dependency=afterok:$probe_dep sbatch_s1_tuned_full40_calibrate.sh)

matched=()
for shard in 0 1 2 3; do
  matched+=("$(sbatch --parsable --dependency=afterok:$cal --export=ALL,PHASE=matched,SHARD_ID=$shard sbatch_s1_tuned_full40_shard.sh)")
done
```

Arm D preserves the six completed probe examples from the original serial job
and resumes only the remaining held-out IDs in four disjoint shards. The
matched phase deliberately covers all 40 held-out IDs because it uses the
newly frozen per-example epsilon map:

```bash
out="$PWD/0_all_combined/reviewer_control_outputs/s1_4khz_full50_seed1"
split="$PWD/0_all_combined/reviewer_control_outputs/S1_SPLIT_MANIFEST.csv"
base="$out/s1/evaluation/bandlimited_4khz/eps0.166_ar0.02_steps1000_seed1_probe/manifest.jsonl"
python 0_all_combined/reviewer_controls/make_id_shards.py --split-csv "$split" \
  --completed "$base" --num-shards 4 --output-prefix "$out/frozen/remaining_probe_shard"
python 0_all_combined/reviewer_controls/make_id_shards.py --split-csv "$split" \
  --num-shards 4 --output-prefix "$out/frozen/all_eval_shard"

dprobe=()
for shard in 0 1 2 3; do
  dprobe+=("$(sbatch --parsable \
    --export=ALL,PHASE=probe,SHARD_ID=$shard,IDS_FILE=$out/frozen/remaining_probe_shard$shard.txt \
    sbatch_s1_4khz_shard.sh)")
done
dprobe_dep=$(IFS=:; echo "${dprobe[*]}")
dcal=$(sbatch --parsable --dependency=afterok:$dprobe_dep sbatch_s1_4khz_sharded_calibrate.sh)
dmatched=()
for shard in 0 1 2 3; do
  dmatched+=("$(sbatch --parsable --dependency=afterok:$dcal \
    --export=ALL,PHASE=matched,SHARD_ID=$shard,IDS_FILE=$out/frozen/all_eval_shard$shard.txt \
    sbatch_s1_4khz_shard.sh)")
done
```

Gate both arms on the per-example 1 dB tolerance; these jobs rerun only
incompatible IDs:

```bash
matched_dep=$(IFS=:; echo "${matched[*]}")
cgate=$(sbatch --parsable --dependency=afterok:$matched_dep sbatch_s1_tuned_snr_gate.sh)
dmatched_dep=$(IFS=:; echo "${dmatched[*]}")
dgate=$(sbatch --parsable --dependency=afterok:$dmatched_dep sbatch_s1_4khz_snr_gate.sh)
sbatch --dependency=afterok:$cgate:$dgate sbatch_s1_baseline_full40_eval.sh
```

All attack commands are resumable. Existing WAVs/manifests are skipped only
within the same immutable run directory; probe and matched artifacts use
different directories.

## Validation

```bash
PYTHONPATH="$PWD/0_all_combined:$PWD" \
  ${WORK_ROOT}/miniconda3/envs/codec-attack/bin/python \
  -m pytest 0_all_combined/reviewer_controls/test_controls.py -q
```
