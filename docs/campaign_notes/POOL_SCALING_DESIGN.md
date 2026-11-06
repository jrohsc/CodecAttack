# Surrogate-pool scaling ladder (reviewer W5)

**Objection.** Every published ensemble run co-optimized `{qwen2_audio, qwen25_omni}`
— N=2, both Qwen-family, both Whisper-encoder. That is a test of Qwen, not a test
of the ensemble hypothesis. Image-domain transfer results that work use 4–8
*diverse* surrogates.

**Design.** Hold one victim out, co-optimize on the first N of a fixed nested
surrogate order, score the held-out victim as N grows. Either transfer emerges at
some N (a real attack result) or it does not (a null that survives "you only tried
two").

## What is actually on this cluster

`${MODEL_ROOT}` is a Whisper monoculture. Ultravox, MiniCPM-o, SALMONN, GLM-4-Voice
and Step-Audio are **absent**. DeSTA2.5 and DIFFA are adapter-only checkpoints
(526 MB / 146 MB) needing backbone assembly. Of what remains, four of six share a
Whisper-large-v3-derived encoder and three share a Qwen2.5 backbone.

| model | encoder | LLM backbone | role |
|---|---|---|---|
| qwen2_audio | whisper-L-v3 | qwen2-7b | surrogate + victim |
| qwen25_omni | whisper-derived | qwen2.5-7b | surrogate + victim |
| qwen25_omni_3b | whisper-derived | qwen2.5-3b | victim only (always held out) |
| audio_flamingo (AF3) | AF-Whisper | qwen2.5-7b | surrogate + victim |
| **phi4** | **Conformer (from scratch)** | phi4-mini-3.8b | surrogate + victim |
| **gemma3n** | **USM** | gemma3n | **surrogate only** |
| kimi_audio | whisper + RVQ | qwen2.5-7b | **victim only** (always held out) |

Only `phi4` and `gemma3n` break the Whisper monoculture. Without them a 5-model
pool converts the objection from "you only tried Qwen" to "you only tried
Whisper" — it does not answer it.

## Asymmetric roles (deliberate)

- **gemma3n is surrogate-only.** `${MODEL_ROOT}` carries only the BASE checkpoints
  (E2B/E4B): no chat template, no instruction tuning. Its gradient path is sound
  (validated below), but asked to transcribe it answers `"1"`. Scoring it as a
  victim would manufacture a 0% that has nothing to do with transfer. It is
  therefore present in **every** rung's pool and never appears as a victim.
- **kimi_audio is victim-only.** Its `compute_loss` raises `NotImplementedError`
  (a differentiable path needs straight-through gradients through its discrete
  RVQ tokenizer). It is scored at every rung as a never-optimized victim, which
  is what makes the ladder's transfer claim rest on a model no rung ever touched.

## Surrogate order

`SURROGATE_ORDER = [qwen2_audio, qwen25_omni, audio_flamingo, phi4, gemma3n]`

Nested, so rung N is exactly rung N−1 plus one model — otherwise "transfer at
N=4" would confound pool size with pool composition. Whisper-lineage first,
non-Whisper encoders last, so any lift specifically attributable to *encoder*
diversity appears as a jump at the phi4 / gemma3n rungs rather than being smeared
across the ladder.

## Gate result: Phi-4 is a surrogate, not a victim

Identical direct waveform PGD (eps=0.05, Adam 2e-3, 300 steps, no EnCodec, no
Opus) on the same carrier/target/prompt:

| model | CE 0 → 300 | target emitted |
|---|---|---|
| qwen2_audio | 4.88 → **0.047** | YES, verbatim |
| gemma3n | 3.65 → **0.030** | YES, verbatim |
| phi4 (bf16) | 3.90 → 2.08 (plateau) | no |
| phi4 (fp32) | 3.91 → 1.85 (plateau) | no |

Phi-4's gradient is real (dense, finite, loss halves) but targeted forcing
plateaus near CE 2, and fp32 does not fix it — so this is not bf16 precision
through the `log` of the 32768-scaled SpeechLib spectrum. Under the
codec-hardened S1 config (eps=1.0, 1000 steps, multi-bitrate Opus EoT) the
white-box pilot scored **0/3 clips at all six Opus bitrates** (16-192 kbps);
1000 steps was no better than 300. The n=50 ceiling job is queued and supersedes
this pilot. (Pilot WERs of 7-9 are inflated: that run predates the `<|end|>`
stop-token fix, so Phi-4 ran past its turn. The `n` -- no target substring --
is the real signal, and the queued ceiling uses the patched wrapper.)

Phi-4 therefore stays a **surrogate** and is dropped as a **victim**: an
unattackable victim yields a 0 that cannot be distinguished from "no transfer".
Its white-box ceiling is still run at n=50 as the published evidence for the
exclusion. Its ensemble contribution should be described as weaker than the
others — its gradient never reaches a successful attack even on itself.

This leaves **gemma3n as the only strong non-Whisper surrogate**, so the
encoder-diversity arm of the ladder rests on it. Do not read the Phi-4 result as
"Conformer encoders are robust" from n=1 model: Phi-4-MM also differs in backbone
size (3.8B) and instruction tuning.

Held-out victims are therefore: `qwen2_audio`, `qwen25_omni`, `audio_flamingo`,
plus the always-held-out `kimi_audio` and `qwen25_omni_3b`. 12 ladder cells.

## Controls (without these the ladder is unreadable)

1. **White-box ceiling per victim** — attack v, evaluate v. A transfer 0 means
   nothing if the attack cannot even land white-box on that victim.
2. **Clean same-prompt control** — unattacked carriers scored under the identical
   bank prompt. The bank prompt alone floors natural speech near 0–10%, so every
   transfer number is read as Δ against this, never as a bare ASR%.
3. **Per-surrogate gradient liveness** — each surrogate must return a finite,
   non-zero gradient through the worker protocol before it joins a rung.

Prompt is identical at optimization and evaluation time, always.

## Correctness gates already passed

- Differentiable front-ends for phi4 (SpeechLib log-fbank → Conformer) and
  gemma3n (USM, 1024-pt FFT magnitude mel, HTK preemphasis) reimplemented in
  `codec_attack/frontends/`, validated against the real HF extractors to
  **3.1e-4** and **1.3e-4** max abs diff, with dense finite gradients.
  A silently-wrong mel yields a healthy-looking gradient and would read as
  "transfer didn't help" — re-run these equivalence tests after any edit.
- All five differentiable surrogates return real gradients through the persistent
  worker protocol in their own conda envs.
- `model_registry.py` is now the single source of truth for name → wrapper. The
  same if/elif chain had been duplicated in four places and had already drifted.
- `ensemble_attack.py` now **hard-fails** on a member it cannot route to
  in-process or subprocess execution. Previously such a member was dropped
  silently and the run rescaled the surviving weights — indistinguishable in the
  results from "adding this surrogate didn't help".

## Cost

50 clips × 1000 steps, ~0.6 s per model-step.

VRAM: every surrogate is resident on the SAME GPU (subprocess workers inherit
`CUDA_VISIBLE_DEVICES`), and an N=3 cell measured ~65 GB. Cells must be submitted
with `--constraint=vram80`; the `gpu` partition is mostly 32-48 GB cards and a
plain `--gres=gpu:1` OOMs part-way through.

| | ladder (16 runs) | + white-box ceilings (4) | total |
|---|---|---|---|
| gradient averaging | 333 GPU-h | 33 GPU-h | **367 GPU-h** |
| model-EoT (1 surrogate sampled/step) | 133 GPU-h | 33 GPU-h | **167 GPU-h** |

Evaluation adds ~2–4 GPU-h (generate-only, 20 bundles × 6 victim scorings).

## Running it

```bash
# attack (codec-attack env; other members auto-route to subprocess workers)
python 0_all_combined/scripts/run_pool_ladder.py --victim audio_flamingo --rung 3 --seed 1

# score, once per victim, inside that victim's own env
<env>/bin/python 0_all_combined/scripts/ensemble_heldout_eval.py \
    --bundle-dir <bundle> --target-model <victim>
```

Both are resumable: SLURM preemption leaves partial bundles that still log
COMPLETED, so the wavs on disk are the only trustworthy progress marker — count
them per cell before scoring.
