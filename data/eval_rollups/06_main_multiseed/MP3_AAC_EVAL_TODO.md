# Cross-scenario codec-robust TODO

Scope: fill `tab:cross_scenario` in `paper/tables.tex` with
**codec-robust** data only (i.e., bundles under this directory:
`0_all_combined/results_codec_robust/`). Non-robust runs under
`0_all_combined/results/` are explicitly out of scope for that table.

## Active chain (2026-04-22 04:27 -- codec eval on 4 complete bundles)

Serial codec-sidecar fill for the bundles whose attack is fully done but
`*_codecs.json` is missing. Driven by
`scripts/chain_codec_eval_2026_04_22.sh`, master log
`scripts/chain_codec_eval_2026_04_22.log` (PID 412390 at launch).

Order (one GPU, env switches between steps 2 and 3):

1. S1 / qwen2_audio / eps=1.5          — env `codec-attack`
2. S2 / qwen2_audio / eps=1.0_mb       — env `codec-attack`
3. S2 / qwen25_omni / eps=1.0_mb       — env `qwen-omni`
4. S3b / qwen25_omni / eps=1.0_mb      — env `qwen-omni`

Prior attack chains (2026-04-21) are all quiescent:
- `chain_all_scenarios_eps1.5.sh` (PID 2580522) aborted 2026-04-21 20:30
  on a stale-path bug in the codec-sidecars stage of S1 qwen2 eps=1.5;
  attack wavs landed fine (25/25), only the sidecar call failed. This
  chain now produces those sidecars as step 1.
- `chain_finish_partials.sh` was alive through 2026-04-21 23:14 and then
  stopped — S3a/qwen25_omni/eps=1.0 attack is at 21/40, not 40/40.

Two pieces per bundle:
1. **Attack run** (training): produces `*_attacked.wav` + `*_summary.json`
   (the latter contains `opus_robustness` rows). Driven by
   `scripts/run_s1_attack_bundleA.py`, `run_s2_attack.py`, `run_s3_attack.py`.
2. **Codec eval** (inference-only): produces `*_codecs.json` sidecars
   (MP3 + AAC-LC at {64,96,128,192}). Driven by
   `scripts/eval_mp3_aac_bundle.py`. Idempotent — skips clips whose
   `*_codecs.json` already exists.

## Env rule
| target_model      | conda env      | python                                                                  |
|-------------------|----------------|-------------------------------------------------------------------------|
| `qwen2_audio`     | `codec-attack` | `${WORK_ROOT}/miniconda3/envs/codec-attack/bin/python` |
| `qwen25_omni`     | `qwen-omni`    | `${WORK_ROOT}/miniconda3/envs/qwen-omni/bin/python`    |
| `audio_flamingo`  | `flamingo3`    | `${WORK_ROOT}/miniconda3/envs/flamingo3/bin/python`    |

The eval loads the target model, so its env must match. Training on one
model cannot share GPU with eval on the same env — queue, don't parallelize.

## State as of 2026-04-22 (disk-verified)

Paths are relative to `0_all_combined/`. `n_adv` = `*_attacked.wav` count;
`codec` = `*_codecs.json` sidecar count. Per-bundle targets: 25 S1, 24 S2,
40 S3a, 20 S3b, ~210 S4.

### S1 finance (`results_codec_robust/01_finance_voice_agent`)
| model           | $\epsilon$ | n_adv  | codec  | status |
|-----------------|-----------:|-------:|-------:|--------|
| qwen2_audio     | 0.5        | 25/25  | 25/25  | **DONE** |
| qwen2_audio     | 1.0        | 25/25  | 25/25  | **DONE** |
| qwen2_audio     | 1.5        | 25/25  |  0/25  | eval running (step 1 of chain) |
| qwen25_omni     | 0.5        | 25/25  | 25/25  | **DONE** |
| qwen25_omni     | 1.0        | 25/25  | 25/25  | **DONE** |
| audio_flamingo  | 0.5        | 25/25  | 25/25  | **DONE** |
| audio_flamingo  | 1.0        | 25/25  | 25/25  | **DONE** |

Each DONE sidecar includes Opus {16,24,32,64,128,192} (via `_summary.json`)
plus MP3 {32,64,96,128,192} and AAC-LC {32,64,96,128,192} (via `_codecs.json`).

### S2 interview (`results_codec_robust/02_interview_screening`)
Robust bundle lives under `eps_{X}_multibitrate/` (script adds suffix when
`--channel-mode multi_bitrate`). The plain `eps_{X}/` folders from the
non-robust driver are out of scope here.

| model           | $\epsilon$ | n_adv  | codec  | status |
|-----------------|-----------:|-------:|-------:|--------|
| qwen2_audio     | 1.0        | 24/24  |  0/24  | eval running (step 2 of chain) |
| qwen25_omni     | 1.0        | 24/24  |  0/24  | eval queued (step 3 of chain) |
| audio_flamingo  | 0.5 / 1.0  | —      | —      | attack TODO |
| qwen2_audio     | 0.5        | —      | —      | attack TODO |
| qwen25_omni     | 0.5        | —      | —      | attack TODO |

### S3a ai-detect (`results_codec_robust/03a_ai_detection_bypass`)
| model           | $\epsilon$ | n_adv  | codec  | status |
|-----------------|-----------:|-------:|-------:|--------|
| qwen2_audio     | 0.5        |  1/40  |  0/1   | attack ❌ stuck (only 1 carrier) |
| qwen2_audio     | 1.0        | 40/40  | 40/40  | **DONE** |
| qwen2_audio     | 1.5        |  4/40  |  0/4   | attack partial |
| qwen25_omni     | 1.0        | 21/40  |  0/21  | attack partial (crashed 2026-04-21 23:14); eval can run on partial |
| audio_flamingo  | 1.0        | 11/40  |  0/11  | attack partial; eval can run on partial |
| qwen25_omni     | 0.5        | —      | —      | attack TODO |
| audio_flamingo  | 0.5        | —      | —      | attack TODO |

### S3b copyright (`results_codec_robust/03b_copyright_bypass`)
| model           | $\epsilon$ | n_adv  | codec  | status |
|-----------------|-----------:|-------:|-------:|--------|
| qwen2_audio     | 0.5        |  1/20  |  0/1   | attack ❌ stuck (only 1 carrier) |
| qwen2_audio     | 1.0        | 20/20  | 20/20  | **DONE** |
| qwen25_omni     | 1.0        | 20/20  |  0/20  | eval queued (step 4 of chain) |
| audio_flamingo  | 1.0        | 12/20  |  0/12  | attack partial; eval can run on partial |
| qwen25_omni     | 0.5        | —      | —      | attack TODO |
| audio_flamingo  | 0.5        | —      | —      | attack TODO |

### S4 sample30 — NO robust bundle yet
| model           | $\epsilon$ | attack | codec eval |
|-----------------|-----------:|--------|------------|
| qwen2_audio     | 0.5 / 1.0  | TODO   | BLOCKED |
| qwen25_omni     | 0.5 / 1.0  | TODO   | BLOCKED |
| audio_flamingo  | 0.5 / 1.0  | TODO   | BLOCKED |

## What's runnable right now (2026-04-22)

**In flight** — 4 codec evals, serial, one GPU
(`scripts/chain_codec_eval_2026_04_22.sh`, PID 412390 at launch):
1. S1 / qwen2_audio / eps=1.5        (25 wavs, `codec-attack`)
2. S2 / qwen2_audio / eps=1.0_mb     (24 wavs, `codec-attack`)
3. S2 / qwen25_omni / eps=1.0_mb     (24 wavs, `qwen-omni`)
4. S3b / qwen25_omni / eps=1.0_mb    (20 wavs, `qwen-omni`)

**Next, once chain clears the GPU**:
1. Codec eval on partial bundles so they can still feed the cross-scenario
   table at a quoted N/target: S3a/qwen25_omni/eps=1.0 (21/40), S3a/audio_flamingo/eps=1.0
   (11/40), S3b/audio_flamingo/eps=1.0 (12/20).
2. Resume attacks on the dead/stuck bundles:
   - S3a/qwen25_omni/eps=1.0 (21/40) — crashed mid-run
   - S3a/audio_flamingo/eps=1.0 (11/40) — partial
   - S3b/audio_flamingo/eps=1.0 (12/20) — partial
   - S3a/qwen2_audio/eps=0.5 and S3b/qwen2_audio/eps=0.5 (both at 1/target — effectively never ran)
   - S3a/qwen2_audio/eps=1.5 (4/40) — partial
3. New attacks: S2 audio_flamingo × {0.5, 1.0}, S2 qwen2_audio/eps=0.5,
   S2 qwen25_omni × {0.5, 1.0}.
4. S4 codec-robust attack sweeps (no bundles exist yet).

## Command pattern

```bash
# from codec_attack/
CA=/work/.../envs/codec-attack/bin/python
OMNI=/work/.../envs/qwen-omni/bin/python
AF3=/work/.../envs/flamingo3/bin/python

# example: codec eval for S1 Omni eps=0.5
$OMNI -u 0_all_combined/scripts/eval_mp3_aac_bundle.py \
    --bundle-dir 0_all_combined/results_codec_robust/01_finance_voice_agent/qwen25_omni/eps_0.5 \
    --target-model qwen25_omni \
    --prompt-text "$(cat 0_all_combined/scenarios/01_finance_voice_agent/system_prompt.txt)"
```
