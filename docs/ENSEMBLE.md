# Multi-victim ensemble attack

This document covers the ensemble experiment: one EnCodec-latent perturbation
co-optimized against three audio LLMs with Opus expectation-over-transformation
(EoT), then evaluated, without adaptation, on the three members and on three
held-out models.

| Role | Model | Download |
|---|---|---|
| Member (primary) | Qwen2.5-Omni-7B | https://huggingface.co/Qwen/Qwen2.5-Omni-7B |
| Member | Audio Flamingo 3 | https://huggingface.co/nvidia/audio-flamingo-3-hf |
| Member | Phi-4-multimodal-instruct | https://huggingface.co/microsoft/Phi-4-multimodal-instruct |
| Held-out | Qwen2-Audio-7B-Instruct | https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct |
| Held-out | Kimi-Audio-7B-Instruct | https://huggingface.co/moonshotai/Kimi-Audio-7B-Instruct (inference code: https://github.com/MoonshotAI/Kimi-Audio) |
| Held-out | Qwen2.5-Omni-3B | https://huggingface.co/Qwen/Qwen2.5-Omni-3B |

No model weights or third-party model code are included in this repository.
Each model is downloaded from the links above under its own license. The
wrappers in `codec_wrappers/models/` only adapt each model to a common
interface (and, for Phi-4, add a differentiable feature front-end,
`codec_wrappers/phi4_frontend.py`).

## 1. Environments

The members need incompatible `transformers` versions, so each runs in its own
conda env and contributes gradients through a persistent worker process
(`scripts/gradient_worker_persistent.py`). Members are routed by env **name**
(`CONDA_ENVS` in `scripts/config.py`), so create the envs with exactly these
names:

| Env | Used for | Setup |
|---|---|---|
| `codec-attack` | driver, Qwen2-Audio | `docs/REPRODUCE.md` §1 |
| `qwen-omni` | Qwen2.5-Omni-7B / 3B | `docs/REPRODUCE.md` §1 |
| `flamingo3` | Audio Flamingo 3 | `docs/REPRODUCE.md` §1 |
| `audio-llm` | Phi-4-multimodal | see below |
| `kimi-audio` | Kimi-Audio (evaluation only) | see below |

```bash
# Phi-4-multimodal: its remote code needs transformers 4.47 / peft 0.11.
conda create -n audio-llm python=3.10
conda activate audio-llm
conda install -c conda-forge ffmpeg
pip install -r requirements_phi4.txt

# Kimi-Audio: follow the install instructions of the Kimi-Audio repository
# (https://github.com/MoonshotAI/Kimi-Audio) in an env named `kimi-audio`,
# then point KIMI_INFER_ROOT at that checkout so `kimia_infer` is importable.
export KIMI_INFER_ROOT=/path/to/Kimi-Audio
```

Every env needs a working `ffmpeg` for the Opus / MP3 / AAC channels. If an
env lacks one, point `FFMPEG_BIN` at any working ffmpeg binary. Kimi-Audio's
`flash-attn` build must match the torch inside `kimi-audio`; if a torch
installed in your user site-packages shadows it, set `PYTHONNOUSERSITE=1`.

As elsewhere, each model path falls back to its Hugging Face repo id; set
`MODEL_PATH_QWEN25_OMNI`, `MODEL_PATH_AUDIO_FLAMINGO_3`, `MODEL_PATH_PHI4_MM`,
`MODEL_PATH_QWEN2_AUDIO`, `MODEL_PATH_KIMI_AUDIO` or `MODEL_PATH_QWEN25_OMNI_3B`
to use a local snapshot instead.

## 2. Running

The attack uses the `jazz_1` music carrier, which is not redistributed. Place
it at `data/music/jazz_1.mp3` and verify it against the SHA-256 in
`docs/MANIFEST_AUDIO.md` §1b.

```bash
SEED=1 bash scripts/run_ensemble_omni_af3_phi4.sh
```

This crafts 224 target strings on the `jazz_1` carrier (4 shards of 56),
evaluates every crafted clip on the six models, clean and after Opus / MP3 /
AAC-LC at 64 and 128 kbps, and writes
`results/<prefix>_seed<SEED>_SUMMARY.{json,md}`. Useful variables:

- `SHARDS="0 1"` runs a subset of shards, so shards can go to separate GPUs.
- `STAGE=craft|eval|summarize` runs a single stage.
- Crafting needs one 80 GB GPU per shard: the three members share it.

Attack settings: epsilon = 10 (EnCodec latent L-inf), 600 steps, step size 0.2,
perceptual weight 0.1, zero initialization, 50 primary-only warmup steps,
multi-bitrate Opus EoT, and `--ensemble-best-after-warmup`, so the saved
perturbation always comes from a step where all three members were in the loss.

After seeds 1, 7 and 42 finish, combine them:

```bash
python scripts/summarize_ensemble_omni_af3_phi_seeds.py
```

## 3. Seeds

All runs use zero initialization, so the seed changes only the random Opus
EoT draws during optimization. Seeds 7 and 42 seed those draws per clip
(`--seed-codec-per-clip`), which makes a clip's trajectory independent of shard
order and resume. Seed 1 predates that flag and did not seed the draws per clip.

## 4. Reference results

`data/eval_rollups/05_ensemble_omni7_af3_phi4/` holds the per-seed and combined
summaries and each shard's run configuration. Strict target-substring success
(%), mean ± sample SD over seeds 1, 7 and 42, n = 224 per seed:

| Role | Victim | Clean | Opus 64 | Opus 128 | MP3 64 | MP3 128 | AAC-LC 64 | AAC-LC 128 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Member | Qwen2.5-Omni-7B | 99.4 ± 0.3 | 99.3 ± 0.3 | 99.4 ± 0.3 | 99.1 ± 0.5 | 99.4 ± 0.3 | 92.7 ± 0.3 | 95.8 ± 1.2 |
| Member | Audio Flamingo 3 | 95.4 ± 0.9 | 94.9 ± 1.0 | 95.2 ± 0.2 | 92.1 ± 2.5 | 95.4 ± 0.9 | 44.3 ± 2.0 | 46.9 ± 1.6 |
| Member | Phi-4-multimodal | 39.6 ± 3.2 | 39.0 ± 0.9 | 40.2 ± 1.6 | 35.6 ± 1.9 | 39.4 ± 1.8 | 27.8 ± 1.7 | 33.9 ± 2.8 |
| Held-out | Qwen2-Audio | 0.9 ± 0.5 | 0.9 ± 0.0 | 0.4 ± 0.5 | 0.4 ± 0.5 | 0.7 ± 0.5 | 5.5 ± 1.4 | 6.1 ± 1.0 |
| Held-out | Kimi-Audio-7B-Instruct | 4.2 ± 2.2 | 4.6 ± 1.4 | 4.2 ± 1.8 | 4.1 ± 2.0 | 3.7 ± 2.3 | 4.5 ± 2.9 | 4.4 ± 2.5 |
| Held-out | Qwen2.5-Omni-3B | 16.1 ± 2.7 | 14.7 ± 1.2 | 15.2 ± 2.8 | 15.2 ± 2.5 | 15.8 ± 3.0 | 13.4 ± 4.8 | 15.2 ± 3.2 |
