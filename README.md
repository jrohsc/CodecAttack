# CodecAttack: Latent-Space Adversarial Attacks on Audio LLMs via Neural Codec Pipelines

Adversarial attacks that operate in EnCodec's continuous latent space to force Audio LLMs into producing attacker-chosen outputs. Uses music as a perceptual carrier to mask perturbation artifacts, enabling covert injection of arbitrary commands into audio-driven autonomous agents.

## Key Contributions

- **Latent-space attacks on Audio LLMs** — prior work targets ASR (Whisper); we attack instruction-following Audio LLMs (Qwen2-Audio, Kimi Audio, Audio Flamingo 3)
- **Music-carrier concealment** — adversarial perturbations are optimized within EnCodec's latent space while preserving perceptual quality via mel-spectrogram loss
- **Codec-robustness evaluation** — attacks survive Opus compression at realistic bitrates (64–128 kbps), relevant to streaming and telephony
- **200-command benchmark** — physical agent commands across 10 categories (navigation, manipulation, safety, delivery, etc.) in two evaluation modes (transcription and compliance)

## Attack Overview

```
                        Latent Space
                    ┌───────────────────┐
Music (MP3/WAV) ──► │ EnCodec Encoder   │──► z_original
                    └───────────────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │  z_adv = z + δ    │  ◄── δ optimized via PGD/Adam
                    │  ‖δ‖∞ ≤ ε         │      (L-inf constraint)
                    └───────────────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │ EnCodec Decoder   │──► adversarial audio (24 kHz)
                    └───────────────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │ Resample 24k→16k  │  (differentiable)
                    └───────────────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │ Audio LLM         │──► target text
                    │ (Qwen2-Audio)     │    "Sure, I will turn left..."
                    └───────────────────┘
```

**Gradient flow**: The entire pipeline from latent perturbation δ through EnCodec decoder, resampling, and the Audio LLM's loss function is differentiable, enabling end-to-end optimization.

## Quick Start (Demo)

```bash
pip install -r requirements.txt
jupyter notebook demo.ipynb
```

The demo notebook (`demo.ipynb`) walks through the full attack pipeline:
1. Load a music carrier (jazz)
2. Run the latent-space attack against Qwen2-Audio (~30 steps to succeed)
3. Listen to original vs adversarial audio
4. Visualize waveforms and spectrograms
5. Test robustness to Opus compression

## Project Structure

```
├── demo.ipynb              # Interactive demo notebook
├── codec_attack.py         # LatentCodecAttacker: core attack pipeline
├── models/
│   ├── base.py             # Abstract audio model interface
│   └── qwen2_audio.py      # Qwen2-Audio wrapper (differentiable loss)
├── attacks/
│   └── latent_codec.py     # EnCodec continuous latent-space wrapper
├── examples/
│   └── audio/              # Example music carrier + pre-generated adversarial audio
└── requirements.txt
```

## Setup

### Prerequisites

- CUDA-capable GPU (>=20 GB VRAM for Qwen2-Audio-7B)
- Python 3.10+
- `ffmpeg` (for Opus robustness testing)

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Model Download

The demo uses Qwen2-Audio-7B-Instruct. Either:

- **HuggingFace Hub** (auto-download): Set `MODEL_PATH = "Qwen/Qwen2-Audio-7B-Instruct"` in the notebook
- **Local download**:
  ```bash
  huggingface-cli download Qwen/Qwen2-Audio-7B-Instruct --local-dir models/Qwen2-Audio-7B-Instruct
  ```

## How It Works

1. **Encode** a music carrier into EnCodec's continuous latent space `z`
2. **Optimize** a perturbation `δ` (bounded by ε in L∞) to minimize:
   - **Behavior loss**: Cross-entropy on target text output
   - **Perceptual loss**: Mel-spectrogram distance to original music
3. **Decode** the perturbed latents `z + δ` back to audio via EnCodec's decoder
4. The gradient flows end-to-end: `δ → decoder → resample → Audio LLM → loss`

## Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `eps` | 0.5 | L-inf bound in EnCodec latent space |
| `alpha` | 0.2 | Adam learning rate |
| `steps` | 150 | Optimization iterations |
| `perceptual_weight` | 0.1 | Weight for mel-spectrogram distance loss |
| `music_duration` | 10.0s | Duration of music carrier |
| `encodec_bandwidth` | 6.0 kbps | EnCodec target bandwidth |

## Technical Notes

- **EnCodec**: `decoder.train()` enables gradient flow; `decoder.eval()` for inference
- **Resampling**: `torchaudio.functional.resample` (24kHz → 16kHz) is differentiable
- **Qwen2-Audio**: Expects 3000-frame mel spectrograms at 16kHz sample rate
- **Audio saving**: Uses `soundfile.write()` (not `torchaudio.save`) due to torchcodec dependency issues
