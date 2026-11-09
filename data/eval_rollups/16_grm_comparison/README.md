# GRM: Utility-Aware Jailbreak Attacks on Audio LLMs via Gradient-Ratio Masking

<p align="left">
  <a href="https://arxiv.org/abs/2604.09222"><img src="https://img.shields.io/badge/Paper-arXiv-b31b1b.svg"></a>
  <a href="https://github.com/159753Fetter/GRM"><img src="https://img.shields.io/badge/Code-GitHub-black"></a>
</p>

This directory includes third-party reference documentation for the GRM baseline. The upstream project accompanies the paper [**GRM: Utility-Aware Jailbreak Attacks on Audio LLMs via Gradient-Ratio Masking**](https://arxiv.org/abs/2604.09222). It covers mask (key-band) generation, universal perturbation training, generation, and evaluation (JSR / WER / RQS / PESQ / STOI) on four target ALLMs: Qwen2-Audio, Qwen2.5-Omni, LLaMA-Omni2, and MERaLiON.

---

## 🔧 Environment Setup

```bash
conda create -n GRM python=3.10
conda activate GRM
pip install -r requirements.txt
```

---

## 📁 Project Structure

```text
GRM/
├── main.py                     # train perturbation tile (Qwen2-Audio, default pipeline)
├── gradient.py                 # computes top-K bands and outputs the mask (Qwen2-Audio)
├── generate.py                 # generation with trained perturbation (Qwen2-Audio)
├── generate_random.py          # random-noise generation (Qwen2-Audio)
├── evaluate_asr.py             # WER evaluation (Qwen2-Audio)
├── evaluate_pesq_stoi.py       # PESQ / STOI perceptual-quality evaluation
├── dataloader.py               # dataset + collate utilities
├── llama3_guard.py             # Llama-Guard-3 based JSR evaluation
├── Llama2/                     # LLaMA-Omni2 specific scripts
├── Qwen2.5/                    # Qwen2.5-Omni specific scripts
├── Meralion/                   # MERaLiON specific scripts
├── AIR-bench/                  # AIR-Bench-Chat inference/scoring pipeline (RQS)
├── logs/                       # stores masks and training checkpoints (Qwen2-Audio)
├── datasets/                   # evaluation datasets for RQS / WER / JSR
├── train.jsonl                 # training data (audio path + prompt)
└── test.jsonl                  # test data for evaluation/inference
```

---

## 📁 Datasets

- **AdvBench-Audio** (`datasets/Explicit_Advbench`, `datasets/advbench.jsonl`): 520 harmful prompts from AdvBench, synthesized to speech with gTTS; split 80/20 into `train.jsonl` (416) / `test.jsonl` (104).
- **LibriSpeech** (`datasets/libri_500`, `datasets/libri_500.json`): first 500 clips from the LibriSpeech test set, used for WER and PESQ/STOI.
- **AIR-Bench-Chat** (`datasets/AIR-Bench-Dataset/Chat`): 800 samples used for RQS.

---

## 🧪 Usage

Root-level scripts run the default Qwen2-Audio pipeline; `<ModelDir>` (`Llama2/`, `Qwen2.5/`, `Meralion/`) scripts follow the same pattern for the other three models.

```bash
# 1) mask generation
python gradient.py --run_batch --topk 48          # or <ModelDir>/mel.py

# 2) train perturbation noise
python main.py --semantic_loss_weight 5           # or <ModelDir>/<model_script>.py, e.g. llama2.py / qwen2.5.py / meralion.py

# 3) generate jailbreak responses
python generate.py                                # or <ModelDir>/response.py

# 4) JSR
python llama3_guard.py

# 5) WER
python evaluate_asr.py                            # or <ModelDir>/evaluate_asr.py

# 6) PESQ / STOI
python evaluate_pesq_stoi.py

# 7) RQS (AIR-Bench-Chat)
python AIR-bench/Inference_Chat.py                # or <ModelDir>/evaluate_rqs.py
python AIR-bench/score_chat.py
python AIR-bench/cal_score.py
```

Number of selected key bands (K) per target model:

| Model | Directory | K |
|---|---|---:|
| Qwen2-Audio | root | 48 |
| Qwen2.5-Omni | `Qwen2.5/` | 128 |
| LLaMA-Omni2 | `Llama2/` | 32 |
| MERaLiON | `Meralion/` | 64 |

> Trained perturbation checkpoints (`*.pt`) are not tracked in this repo; only the precomputed key-band masks (`*.npz`) are included. Paths such as `--model_path`, `--mask_path`, `--save_dir` are placeholders — override them to match your own environment.

---

## 🙏 Acknowledgments

This repository builds upon the following publicly available models and benchmarks:

- [Qwen2-Audio-7B-Instruct](https://huggingface.co/Qwen/Qwen2-Audio-7B)
- [Qwen2.5-Omni-7B](https://huggingface.co/Qwen/Qwen2.5-Omni-7B)
- [MERaLiON-2-10B](https://huggingface.co/MERaLiON/MERaLiON-2-10B)
- [LLaMA-Omni2-7B](https://huggingface.co/ICTNLP/LLaMA-Omni2-7B)
- [Llama-Guard-3-8B](https://huggingface.co/meta-llama/Llama-Guard-3-8B)
- [Whisper-large-v3](https://huggingface.co/openai/whisper-large-v3)
- [AdvBench](https://huggingface.co/datasets/walledai/AdvBench)
- [LibriSpeech](https://www.openslr.org/12)
- [AIR-Bench](https://github.com/OFA-Sys/AIR-Bench)
