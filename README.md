# Codec-Robust Attacks on Audio LLMs (anonymous code and data)

Reviewer-requested artifacts, reproduction commands, completed results, and remaining limitations: [Master checklist](docs/REBUTTAL_MASTER_CHECKLIST.md).

This is the anonymous release that backs the paper *Codec-Robust Attacks on Audio LLMs*. It contains the attack code, the codec wrappers, the carrier dataset, the per-pair target-text mapping, and every JSON / JSONL eval rollup used to produce the tables and figures. Adversarial (perturbed) audio is **not** included; reviewers can regenerate it from the carriers using the included scripts. The paper PDF is submitted separately through the conference review system.

## Contents

```
codec_wrappers/            Continuous-latent encode/decode wrappers
    encodec_wrapper.py     EnCodec (24 kHz, 75 fps, 128-d latent)
    mimi_wrapper.py        Kyutai Mimi (24 kHz, 12.5 fps, 512-d latent)
    dac_wrapper.py         Descript Audio Codec (24 kHz, 75 fps, 1024-d latent)
    snac_wrapper.py        SNAC multi-scale RVQ
    channel_augmentation.py    Differentiable Opus / MP3 / AAC straight-through proxies
    phi4_frontend.py       Differentiable Phi-4-multimodal audio feature extractor
    models/                Target-LLM adapters (Qwen2-Audio, Qwen2.5-Omni,
                           Audio Flamingo 3, Phi-4-multimodal, Kimi-Audio,
                           PersonaPLEX/Moshi); no third-party model code

scripts/                   Attack drivers, eval, table builders
    config.py                   Centralized paths, target maps, prompt modes
    check_env.py                Smoke check: run this first on a fresh clone
    latent_attack.py            Core latent-space PGD
    robust_latent_attack.py     With codec EoT
    ensemble_attack.py          Multi-model ensemble PGD
    model_registry.py           Model name -> adapter (shared by driver and workers)
    gradient_worker*.py         Per-env gradient workers for ensemble members
    ensemble_heldout_eval.py    Member / held-out evaluation of ensemble audio
    run_ensemble_omni_af3_phi4.sh   Omni-7B + AF3 + Phi-4 ensemble: craft, eval, summarize
    aggregate_extra_heldout.py, summarize_ensemble_*.py   Ensemble rollups
    music_carrier.py            Carrier loading + mel perceptual distances
    aac_channel.py              AAC / MP3 straight-through channel
    demo_ota.py                 Over-the-air channel simulation
    realistic_channel.py        Speaker nonlinearity + physical channel filters
    watermark_defense.py        AudioSeal defense evaluation
    run_benchmark.py            Basic (non-robust) benchmark driver
    run_robust_benchmark.py     Main benchmark driver (EnCodec)
    run_mimi_full.py            Mimi cross-codec attack
    run_dac_full.py             DAC cross-codec attack
    eval_models.py              Cross-model evaluation helpers
    eval_quality_*.py           SNR / LSD / PESQ / dLUFS audio quality
    eval_dac_partial19.py       Routing + eval over the 19 DAC attacked wavs
    build_*.py                  Aggregation scripts that turn raw JSONs into the paper tables
    campaigns/                  The raw campaign layer, as it ran on our cluster:
                                SLURM launchers, chain/sweep drivers, and the
                                analysis scripts behind the mechanistic figures.
                                Absolute paths are placeholder-substituted, so
                                treat these as a record of what was run rather
                                than drop-in runnable (see docs/RESULTS_MAP.md).
    reviewer_controls/          Band-limited / SNR-matched control harness

data/
    speech/                     TTS-generated carriers (Microsoft Edge TTS)
        01_finance/             S1: 50 banking voice carriers, target text, TTS scripts
        02_interview/en/        S2 English: 25 carriers, target text, TTS scripts
        02_interview/zh/audio/  S2 Mandarin: 18 TTS carriers (chinese_bad, chinese_murmur, chinese_v2)
        02_interview/           S2 target maps (full Mandarin pairing)
        03b_copyright/          S3b metadata (audio not redistributed)
        04_podcast_advert/      S4: 4 podcast / advertisement carriers + scripts
    music/
        ai_generated/           S3a: 8 Suno-generated music carriers
        03a_ai_detection_bypass/    S3a target-text mapping
        03b_copyright_bypass/       S3b target-text mapping (audio not redistributed)
    eval_rollups/               Per-cell ASR + audio-quality numbers (no audio).
                                docs/RESULTS_MAP.md maps every paper table and
                                figure to the directory that backs it.
        01_finance_voice_agent/         S1  (single-seed, original submission)
        02_interview_screening/         S2
        03a_ai_detection_bypass/        S3a
        03b_copyright_bypass/           S3b (cross-codec results live here)
            encodec_eps1.0/, mimi_eps0.2/, dac_eps0.6194/
        04_simulated_ota_physical_agent/    S4
        05_ensemble_omni7_af3_phi4/     Omni-7B + AF3 + Phi-4 ensemble (seeds 1, 7, 42)
        06_main_multiseed/              Main campaign: eps sweep x 4 scenarios x
                                        3 models, seeds {1,7,42}. Backs the main
                                        tables and the seed grid.
        07_eot_train_codec/             EoT training-proxy matrix: opus / mp3 /
                                        aac / ensemble(Opus+AAC) / encodec / dac
        08_non_robust/                  No-EoT ablation (clean-channel objective)
        09_waveform_pgd/                Waveform-PGD baseline + n=50 matched-rate arm
        10_ensemble_victim_s1/          Ensemble-victim co-optimization, S1
        11_ensemble_transcribe/         Qwen2-Audio + Omni-7B + AF3 ensemble,
                                        held-out incl. DeSTA2.5
        12_capacity_ablation/           Target length / duration capacity cliff
        13_mechanistic/                 Per-Bark energy, decoder frequency response,
                                        encoder residual, cosine-through-codec
        14_neural_resynthesis/          DAC / Mimi / SNAC re-encode defense
        15_reviewer_controls/           Band-limited and SNR-matched controls
        16_grm_comparison/              GRM reproduction + judge audit
        _results_watermark/             AudioSeal defense
        _results_watermark_channel_sweep/   AudioSeal, per-delivery-channel
        _results_waveguard/             WaveGuard defense
        _results_waveform/              Waveform-PGD baseline (original submission)

configs/                   Pipeline YAML
scenarios/                 Per-scenario task prompts (user role), scenario.json, target maps
target_drafts/             Per-scenario target-command JSONs

docs/
    RESULTS_MAP.md              Paper table / figure -> rollup path, the scoring
                                convention, and a snippet that recomputes a
                                published cell from the shipped JSON alone
    MANIFEST_AUDIO.md           SHA-256 manifest of the not-redistributed carriers (copyrighted music + MagicData clips)
    REPRODUCE.md                Environment setup, attack, eval, table-rebuild instructions
    ENSEMBLE.md                 Multi-victim ensemble attack: envs, model downloads, run, results
    campaign_notes/             Working notes and inventories for the campaigns above
```

## Cohort configuration and current extensions

`configs/paper_cohorts.json` records the original paper cohort: 50 banking recordings, 25 English interview recordings, 8 AI-music recordings (40 pairs), and 9 copyright recordings (45 pairs). Task text is passed in the user message alongside audio. The legacy `system_prompt.txt` filenames do not imply system-role placement.

The independently composed content extension is documented separately in `data/reviewer_update/content_extension/`; its evaluation is incomplete. It does not replace original datasets or alter frozen run configurations. Historical campaign scripts remain provenance records; their old menu counts and status comments are not the authoritative cohort inventory.

## Quick reproduction

```bash
conda create -n codec-attack python=3.10 && conda activate codec-attack
conda install -c conda-forge ffmpeg
pip install -r requirements.txt
python scripts/check_env.py        # verifies imports, paths, data, and ffmpeg
```

1. Install dependencies as above (details in `docs/REPRODUCE.md`).
2. Optionally export model paths (`MODEL_PATH_QWEN2_AUDIO`, `MODEL_PATH_QWEN25_OMNI`, `MODEL_PATH_AUDIO_FLAMINGO_3`) to use local snapshots. If unset, the scripts fall back to the public HuggingFace repo ids and download on first use.
3. For S3a / S3b only: obtain the 9 commercial-music carriers through legitimate means, place at `data/music/copyrighted/`, verify with the SHA-256 manifest. (TTS carriers and Suno music are already in this repo.)
4. Run an attack driver: e.g. `python scripts/run_dac_full.py --mode full`.
5. Regenerate the paper's tables: `python scripts/build_main_results_tables.py`.

For the multi-victim ensemble attack (Qwen2.5-Omni-7B + Audio Flamingo 3 +
Phi-4-multimodal, evaluated on three held-out models), see `docs/ENSEMBLE.md`.

Attacked audio and any newly computed metrics are written under `results/`; the
reference rollups shipped in `data/eval_rollups/` are never overwritten.

## What is in `carrier_to_target.json`

Each scenario's `carrier_to_target.json` lists, for every (carrier, target) pair the paper attacks:

- the carrier filename (relative to the scenario's `audio/` folder)
- the target text the attack tries to make the model output (the "command" or induced response)
- the carrier's clean transcript (when applicable)
- the category bucket the pair belongs to

Concretely, the paper's $n{=}45$ S3b experiments map 9 copyrighted music carriers to 5 copyright-bypass target strings; that mapping is `data/music/03b_copyright_bypass/carrier_to_target.json`.

## What is not included, and why

- **Adversarial / perturbed audio.** ~10k WAVs across all scenarios; regenerable from the carriers with the included attack drivers.
- **Copyrighted music carriers** (9 commercial recordings used as S3b and partial S3a carriers). Not ours to redistribute. `docs/MANIFEST_AUDIO.md` lists the SHA-256 hashes so reviewers can verify the exact files we used after obtaining them locally.
- **MagicData Mandarin clips** (6 of the 24 Mandarin S2 carriers). License the corpus from Magic Data Technology Co., Ltd. The remaining 18 Mandarin carriers (`chinese_bad_*`, `chinese_murmur_*`, `chinese_v2_*`) are TTS-generated and included.
- **Target-LLM model weights and third-party model code.** Download from HuggingFace under the model owners' licenses (`Qwen/Qwen2-Audio-7B-Instruct`, `Qwen/Qwen2.5-Omni-7B`, `Qwen/Qwen2.5-Omni-3B`, `nvidia/audio-flamingo-3-hf`, `microsoft/Phi-4-multimodal-instruct`, `moonshotai/Kimi-Audio-7B-Instruct`). Kimi-Audio's inference code comes from https://github.com/MoonshotAI/Kimi-Audio. Links and setup are in `docs/ENSEMBLE.md`.

## Anonymity

Author-specific names, affiliations, account links, and machine paths have been removed from the released files. Machine paths use environment-variable placeholders (`${REPO_ROOT}`, `${MODEL_PATH_*}`, etc.). Third-party citations, dataset and model names, TTS voice IDs, and names in experimental prompts and responses remain part of the research artifacts. Incidental institution mentions in one saved response file are explicitly redacted; its recorded scores are preserved, and its metadata identifies the affected fields. Git commit identifiers are omitted from the exported provenance; content hashes identify the released artifacts. This statement covers file contents, not the hosting account or Git history.

## License

This repository is released for the purpose of double-blind review. After deanonymization, code will be released under MIT and data under CC-BY-4.0. The copyrighted music carriers (which are not in this repo) remain subject to their respective rights holders' licenses.
