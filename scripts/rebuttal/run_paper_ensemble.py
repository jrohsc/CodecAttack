"""Paper-described ensemble/optimizer settings at epsilon one on domain cohorts.

The exact pre-update state scored by the eligible training objective is saved.
The legacy returned state is retained separately to expose the old one-step
scored/saved discrepancy; no held-out response selects either checkpoint.
"""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from rebuttal.cohort import validate_cohort
from rebuttal.run_followup import save

MEMBERS = ['qwen25_omni', 'audio_flamingo', 'phi4']


def settings(smoke=False, epsilon=1.):
    return dict(members=MEMBERS, epsilon=epsilon, alpha=.2, steps=4 if smoke else 600,
        warmup=1 if smoke else 50, perceptual_weight=.1, initialization='zeros',
        grad_normalize=True, weights='equal', channel_warmup_ratio=.3,
        training_opus_rates=[16,24,32,64,128], checkpoint='lowest eligible training objective, exact scored state',
        channels=['clean','opus64','opus128','mp3_64','mp3_128','aac_64','aac_128'], smoke=smoke)


def validate_history(history, spec, snapshot):
    import numpy as np
    for m in MEMBERS:
        expected = spec['steps'] - (spec['warmup'] if m != MEMBERS[0] else 0)
        losses = history['per_model_loss'][m]
        if len(losses) != expected or not np.isfinite(losses).all():
            raise RuntimeError(f'Missing/nonfinite contribution from {m}')
    start = max(spec['warmup'], int(spec['steps'] * spec['channel_warmup_ratio']))
    if snapshot['step'] < start + 1 or not math.isfinite(snapshot['loss']):
        raise RuntimeError('Selected checkpoint is not eligible')
    eligible = history['loss'][start:]
    if abs(snapshot['loss'] - min(eligible)) > 1e-6 * max(1, abs(min(eligible))):
        raise RuntimeError('Selected checkpoint is not the minimum eligible objective')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cohort', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, choices=[1,7,42], required=True)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--eps', type=float, choices=[1,10], default=1)
    p.add_argument('--released-checkpoint', action='store_true')
    p.add_argument('--shard-id', type=int, default=0)
    p.add_argument('--num-shards', type=int, default=1)
    a = p.parse_args()
    cohort = validate_cohort(a.cohort)
    if a.num_shards < 1 or not 0 <= a.shard_id < a.num_shards:
        raise ValueError('Invalid shard')
    pairs = cohort['pairs'][a.shard_id::a.num_shards]
    pairs = pairs[:1] if a.smoke else pairs
    out = a.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    lock = (out/'run.lock').open('w'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    spec = dict(**settings(a.smoke, a.eps), scenario=cohort['scenario'], seed=a.seed,
                expected_examples=len(pairs), cohort_sha256=hashlib.sha256(a.cohort.read_bytes()).hexdigest(),
                shard_id=a.shard_id, num_shards=a.num_shards,
                codec_seed_rule='SHA256(seed:jazz_1:command_id), first 8 bytes, torch seed' if any('paper_command_id' in p for p in pairs) else 'seed reset per example',
                difference_from_paper=cohort.get('difference_from_paper', 'epsilon=1; domain carriers/targets/prompts; per-example seeded draws for all seeds; scored-state checkpoint alignment'))
    if a.released_checkpoint:
        spec['checkpoint'] = 'released returned post-update state at lowest eligible objective'
        spec['check_every'] = 25
    cp=out/'config.json'
    if cp.exists() and json.loads(cp.read_text()) != spec:
        raise RuntimeError('Run configuration changed')
    save(cp,spec)
    manifest=out/'records.jsonl'
    rows=[json.loads(x) for x in manifest.read_text().splitlines()] if manifest.exists() else []
    done={r['example_id'] for r in rows}
    if len(done)!=len(rows) or not done <= {p['example_id'] for p in pairs}:
        raise RuntimeError('Invalid completed records')
    for row in rows:
        for name,sha in row['audio_hashes'].items():
            if hashlib.sha256(Path(row['checkpoints']['selected'][name]).read_bytes()).hexdigest()!=sha:
                raise RuntimeError('Completed audio changed')
    if len(done)==len(pairs): return
    import config
    import numpy as np
    import torch
    from ensemble_attack import EnsembleLatentAttacker
    from music_carrier import load_music
    from reviewer_controls.run_attack import seed_all
    from reviewer_controls.common import write_pcm16_pair
    seed_all(a.seed)
    attacker=EnsembleLatentAttacker(ensemble_models=MEMBERS, ensemble_weights={m:1 for m in MEMBERS},
        in_process_models=['qwen25_omni'], subprocess_models=['audio_flamingo','phi4'], primary_model='qwen25_omni',
        eps=a.eps, alpha=.2, warmup_steps=spec['warmup'], perceptual_weight=.1, channel_mode='multi_bitrate',
        multi_bitrate_kbps=[16,24,32,64,128], channel_warmup_ratio=.3, best_after_warmup=True,
        strict_workers=True, grad_normalize=True, rand_init=False, seed=a.seed, verbose=True, diagnostic_interval=50)
    try:
        for pair in pairs:
            eid=pair['example_id']
            if eid in done: continue
            seed_all(a.seed)
            if 'paper_command_id' in pair:
                identity=f"{a.seed}:jazz_1:{pair['paper_command_id']}".encode()
                torch.manual_seed(int.from_bytes(hashlib.sha256(identity).digest()[:8], 'big'))
            audio=load_music(pair['carrier_path'],duration=pair['duration_s'])
            start=time.perf_counter()
            result=attacker.attack(audio,pair['target'],steps=spec['steps'],check_every=25 if a.released_checkpoint else 0,
                                   prompt=pair['prompt'],music_name=Path(pair['carrier']).stem)
            snap=attacker.diagnostic_checkpoints['eligible']
            validate_history(result.history,spec,snap)
            bound=float(snap['delta'].abs().max())
            if not math.isfinite(bound) or bound>a.eps+1e-5: raise RuntimeError('Invalid latent bound')
            with torch.no_grad():
                wave=attacker.codec.decode_from_continuous((result.original_latents+snap['delta']).to(attacker.device)).float().cpu().squeeze().numpy()
            if not np.isfinite(wave).all(): raise RuntimeError('Nonfinite audio')
            if a.released_checkpoint:
                wave=result.adversarial_wav.detach().float().cpu().squeeze().numpy()
                bound=float(result.latent_perturbation.abs().max())
                if not math.isfinite(bound) or bound>a.eps+1e-5: raise RuntimeError('Invalid released latent bound')
            metrics=write_pcm16_pair(out/'audio',eid,audio.squeeze().numpy(),wave,24000)
            legacy=write_pcm16_pair(out/'legacy_audio',eid,audio.squeeze().numpy(),result.adversarial_wav.detach().float().cpu().squeeze().numpy(),24000)
            row=dict(**pair,config=spec,selected_checkpoint='selected',elapsed_s=time.perf_counter()-start,
                checkpoints={'selected':dict(**metrics,step=snap['step'],training_loss=snap['loss'],delta_linf=bound,
                                 loss_audio_aligned=not a.released_checkpoint),
                             'legacy_returned':legacy},
                audio_hashes={k:hashlib.sha256(Path(metrics[k]).read_bytes()).hexdigest() for k in ['adv_path','carrier_ref_path']},
                member_contribution_counts={m:len(result.history['per_model_loss'][m]) for m in MEMBERS})
            save(out/(eid+'_history.json'),result.history)
            with manifest.open('a') as f:
                f.write(json.dumps(row)+'\n');f.flush();os.fsync(f.fileno())
            done.add(eid)
            save(out/'status.json',dict(completed=len(done),expected=len(pairs),job=os.getenv('SLURM_JOB_ID')))
            print('COMPLETED',eid,len(done),len(pairs),flush=True)
        save(out/'completion.json',dict(complete=True,n=len(done),smoke=a.smoke))
    finally: attacker.close_workers()


if __name__=='__main__': main()
