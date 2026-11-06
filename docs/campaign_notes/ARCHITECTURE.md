# EVA-Audio architecture

## Upstream EVA components — disposition

| Upstream component | Decision |
|---|---|
| `user_simulator/` (ElevenLabs) | **Removed.** Replaced by pre-crafted adversarial wavs on disk. |
| `assistant/pipeline/` (Pipecat) | **Removed.** Replaced by direct local model inference (`adapters/`). |
| `assistant/services/` (STT/TTS factories) | **Removed.** Audio LLMs are audio-native — no STT/TTS needed. |
| `orchestrator/runner.py` (WebSocket) | **Replaced.** Our `runner.py` is a plain Python loop. |
| `models/` (pydantic data models) | **Kept in spirit**, reimplemented as dataclasses in `src/eva_attack/models.py`. |
| `metrics/` engine | **Kept in spirit.** Own minimal metric registry under `metrics/`. |
| `configs/prompts/` (YAML judge rubrics) | **Kept.** Same pattern under `configs/prompts/`. |
| `data/*.jsonl` scenarios | **Kept in spirit.** Our layout is one dir per scenario, not one giant JSONL. |

## Data flow

```
scenarios/NN_*/           scenario.json + system_prompt.txt + target.txt + audio/
        │
        ▼  scenario_loader.load_scenario
   Scenario dataclass
        │
        ▼  runner.run_scenario
  for audio × channel × model:
      channel.apply(audio) -> routed_audio
      adapter.respond(system_prompt, routed_audio, history, tools) -> output
      for metric in scenario.metrics: metric(output, scenario) -> MetricResult
      emit RunResult -> results.jsonl
        │
        ▼
   scenarios/NN_*/results/results.jsonl
```

## Why scenario-per-subfolder (vs single JSONL)

1. Each scenario carries a lot of data: a long system prompt, a target file, a directory of audio, a README. Easier to diff + review one at a time.
2. Lets us symlink `audio/` to existing `1_results_qa/`, `1_results_transcribe/`, etc. without copying GB of wavs.
3. Natural mapping to how the existing `results_*/` dirs are already organized.
4. Good git hygiene — individual scenarios can be added or revised in small PRs.

## Extension points

- **New category** → add `scenarios/NN_<name>/` with a fresh category string.
- **New channel** → add `channels/<name>.py` + register in `channels/__init__.py:get_channel`.
- **New model target** → add `adapters/<name>.py` + register in `adapters/__init__.py:get_adapter`.
- **New metric** → add function in `metrics/<file>.py` + register in `metrics/__init__.py:get_metric`.

## Dependencies outside this folder

Parent `codec_attack/` provides:
- `channel_augmentation.py` (Opus/AAC/noise simulation — reused by `channels/opus.py` and `channels/physical_sim.py`)
- `generate_rirs.py` (RIR generator)
- `compute_pesq.py`, `compute_wer.py` (perceptual + WER)
- `external/whisper-inject-v2-fork/models/*.py` (Qwen2-Audio, Audio Flamingo, Qwen2.5-Omni wrappers)
- `external/whisper-inject-v2-fork/core/judge.py` (LLM judge)

Scripts add those to `PYTHONPATH` via `export PYTHONPATH="$PWD/src:$PWD/..:..."`.
