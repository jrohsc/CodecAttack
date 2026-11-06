#!/usr/bin/env python3
"""Summarize the published seed-1 bundle with zero-init seeds 7 and 42."""
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / 'results'
BASE = 'benchmark_ensemble3_opuseot_transcribe_eps10.0_jazz_1_bestpostcodec_steps600_zeroeot'
PUBLISHED_BASE = 'benchmark_ensemble3_opuseot_transcribe_eps10.0_jazz_1_bestpostcodec_steps600'
SEEDS = (1, 7, 42)
CHANNELS = ('clean', 'opus64', 'opus128', 'mp3_64', 'mp3_128', 'aac_64', 'aac_128')
MODELS = (
    ('qwen2_MEMBER', 'Qwen2-Audio', 'member'),
    ('omni7b_MEMBER', 'Qwen2.5-Omni-7B', 'member'),
    ('af3_MEMBER', 'Audio Flamingo 3', 'member'),
    ('omni3b_HELDOUT', 'Qwen2.5-Omni-3B', 'held out'),
    ('kimi_HELDOUT_100tok', 'Kimi-Audio-7B-Instruct', 'held out'),
    ('phi4_HELDOUT', 'Phi-4-multimodal', 'held out'),
)

def main():
    by_seed = {}
    snr_by_seed = {}
    for seed in SEEDS:
        prefix = PUBLISHED_BASE if seed == 1 else f'{BASE}_seed{seed}'
        path = ROOT / f'{prefix}_AGGREGATED.json'
        core = json.loads(path.read_text())
        assert core['n'] == 224 and core['channels'] == list(CHANNELS), path
        by_seed[seed] = {model: {ch: core['counts'][model][ch]['exact'] for ch in CHANNELS}
                         for model, _, _ in MODELS[:4]}
        for model, _, _ in MODELS[4:]:
            extra_path = ROOT / f'{prefix}_{model}_AGGREGATED.json'
            extra = json.loads(extra_path.read_text())
            expected_model = 'kimi_audio' if model.startswith('kimi') else 'phi4'
            assert extra['n'] == 224 and extra['heldout_model'] == expected_model, extra_path
            by_seed[seed][model] = extra['counts']
        snrs = []
        for shard in range(4):
            bundle = ROOT / f'{prefix}_shard{shard}of4'
            for record_path in (bundle / 'qwen2_audio').glob('*/record.json'):
                record = json.loads(record_path.read_text())
                snrs.append(float(record['direct']['snr_db']))
        assert len(snrs) == 224, (seed, len(snrs))
        snr_by_seed[str(seed)] = round(statistics.mean(snrs), 3)
    summary = {'seeds': list(SEEDS), 'n_per_seed': 224,
               'attack': '3-member Opus-EoT, epsilon=10, 600 steps, zero initialization',
               'opus_rng': {'1': 'published run: no explicit per-clip seed',
                            '7': 'deterministic per-clip seed',
                            '42': 'deterministic per-clip seed'},
               'metric': 'case-insensitive exact target substring',
               'channels': list(CHANNELS), 'models': {},
               'mean_waveform_snr_db_by_seed': snr_by_seed,
               'mean_waveform_snr_db_all': round(statistics.mean(snr_by_seed.values()), 3)}
    lines = ['# 224-target Opus-EoT runs 1, 7, and 42', '',
             'The published zero-initialized seed-1 bundle is combined with new zero-initialized runs 7 and 42. The published run did not explicitly seed Opus draws per clip; runs 7 and 42 do. Values are strict target-substring ASR (%), mean ± sample SD over these three runs, with 224 targets per run.',
             f"Mean waveform SNR across the sweep: {summary['mean_waveform_snr_db_all']:.2f} dB; per-seed means: {snr_by_seed}.", '',
             '| Role | Victim | Clean | Opus 64 | Opus 128 | MP3 64 | MP3 128 | AAC-LC 64 | AAC-LC 128 |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for key, label, role in MODELS:
        cells = []
        per_channel = {}
        for ch in CHANNELS:
            vals = [100 * by_seed[seed][key][ch] / 224 for seed in SEEDS]
            per_channel[ch] = {'percentages': [round(x, 1) for x in vals],
                               'mean_pct': round(statistics.mean(vals), 1),
                               'sample_sd_pct': round(statistics.stdev(vals), 1)}
            cells.append(f"{per_channel[ch]['mean_pct']:.1f} ± {per_channel[ch]['sample_sd_pct']:.1f}")
        summary['models'][key] = {'label': label, 'role': role,
                                  'per_seed_counts': {str(seed): by_seed[seed][key] for seed in SEEDS},
                                  'channels': per_channel}
        lines.append('| ' + ' | '.join((role, label, *cells)) + ' |')
    json_path = ROOT / f'{PUBLISHED_BASE}_RUNS_1_7_42_SUMMARY.json'
    md_path = ROOT / f'{PUBLISHED_BASE}_RUNS_1_7_42_SUMMARY.md'
    json_path.write_text(json.dumps(summary, indent=2) + '\n')
    md_path.write_text('\n'.join(lines) + '\n')
    print(json_path)
    print(md_path)

if __name__ == '__main__':
    main()
