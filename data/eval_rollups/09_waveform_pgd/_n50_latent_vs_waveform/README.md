# n=50 latent vs. waveform comparison

Scales the S1 latent-vs-waveform table (`paper-neurips/tables/tab_latent_vs_waveform.tex`)
from n=21 to n=50 by doubling the S1 carrier pool and re-running both attacks
on the same 50 (carrier, target) pairs.

## What was changed in the repo

1. `data/speech/01_finance/generate.py` — added 5 new EdgeTTS voices
   (`ana`, `steffan`, `liam`, `emily`, `thomas`). Re-running `generate.py`
   is idempotent (skips wavs that already exist).
2. `data/speech/01_finance/audio/` — now holds 50 carriers
   (5 scripts × 10 voices), up from 25.
3. `data/speech/01_finance/carrier_to_target.json` — extended to 50 pairs.
   Each new carrier has its own unique target sentence inside its
   script-anchored category (5 new targets per category). Existing 25
   pairs are unchanged.

## What still needs to run (GPU work)

Two attack bundles must be brought up to the full 50 pairs each:

| Bundle | Path | Pairs done before | Pairs to run |
|---|---|---|---|
| Latent eps=1.0 | `results_codec_robust/01_finance_voice_agent/qwen2_audio/eps_1.0/` | 25 | 25 (the 5 new voices) |
| Waveform eps=0.075 | `results_codec_robust_waveform/01_finance_voice_agent/qwen2_audio/eps_0.075/` | 21 | 29 (4 missing q3 + 25 new) |

Per-pair runtime, observed on the existing bundles (A100 80 GB):

- Latent attack ≈ 6.5 min (1000 PGD steps, multi-bitrate Opus EOT).
- Waveform attack ≈ 8.7 min (same step budget, Opus EOT in waveform space).
- MP3+AAC sidecar eval ≈ 30 s per pair.

So the remaining work is roughly **2.7 h latent + 4.2 h waveform**.

## How to run

The two attacks live in independent on-disk bundles, so they're safe to
launch from two terminals as long as the GPU has memory to spare.
Both scripts are **resumable** — Ctrl-C and re-run is fine; each pair's
adversarial wav and per-codec sidecar JSON is created atomically and
skipped on the next pass.

```bash
# Terminal A — latent attack + MP3/AAC sidecar
bash 0_all_combined/n50_latent_vs_waveform/run_latent.sh

# Terminal B — waveform attack + MP3/AAC sidecar
bash 0_all_combined/n50_latent_vs_waveform/run_waveform.sh

# Once enough pairs are in BOTH bundles, rebuild the LaTeX table:
bash 0_all_combined/n50_latent_vs_waveform/rebuild_table.sh
```

`rebuild_table.sh` always intersects the two bundles, so it works even
when the runs are partially complete — the table just reports n equal
to whatever overlap exists at that moment.

### GPU note

The cluster's only A100 (80 GB) is enough to run a single attack
comfortably (Qwen2-Audio + EnCodec + PGD grads ≈ 30–40 GB per process).
Running both bash files on the same GPU will work in principle but may
hit OOM under unfortunate scheduling. If that happens, run them
sequentially (waveform first, then latent — or vice versa). Since both
are resumable, an OOM kill is harmless.

If a second GPU becomes available, set
`CUDA_VISIBLE_DEVICES=1 bash run_waveform.sh` so the two runs don't
share memory.

## Helper scripts in this folder

- `extend_pairs.py` — appended the 25 new pairs to
  `carrier_to_target.json`. Idempotent; safe to re-run.
- `run_latent.sh` — launches latent attack + sidecar eval.
- `run_waveform.sh` — launches waveform attack + sidecar eval.
- `rebuild_table.sh` — regenerates the markdown roll-up,
  `paper/tables/s1_latent_vs_waveform_codecs.tex`, and the
  `paper-neurips/tables/tab_latent_vs_waveform.tex` copy.
- `logs/` — tee'd stdout from each stage for after-the-fact inspection.
