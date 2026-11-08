"""Excluded-model transfer or objective quality for completed domain cells."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', type=Path, required=True)
    ap.add_argument('--mode', choices=['transfer', 'model', 'quality'], required=True)
    ap.add_argument('--model', choices=['qwen2_audio', 'qwen25_omni', 'audio_flamingo', 'phi4', 'qwen25_omni_3b', 'kimi_audio'], default='audio_flamingo')
    a = ap.parse_args()
    spec = json.loads((a.input / 'config.json').read_text())
    rows = [json.loads(x) for x in (a.input / 'records.jsonl').read_text().splitlines()]
    if spec['smoke'] or len(rows) != spec['expected_examples'] or len({r['example_id'] for r in rows}) != len(rows):
        raise ValueError('Incomplete cohort or smoke experiment')
    dest = a.input / ((a.model if a.mode == 'model' else a.mode) + '.jsonl')
    done = {json.loads(x)['example_id'] for x in dest.read_text().splitlines()} if dest.exists() else set()
    import config
    import numpy as np
    import soundfile as sf
    from reviewer_controls.common import codec_roundtrip, snr_db, target_match
    model = None
    cache = {}
    for row in rows:
        eid = row['example_id']
        if eid in done:
            continue
        final = row['checkpoints'][row.get('selected_checkpoint', 'final')]
        for key, sha in row['audio_hashes'].items():
            if hashlib.sha256(Path(final[key]).read_bytes()).hexdigest() != sha:
                raise ValueError('Audio changed since optimization')
        ref, sr = sf.read(final['carrier_ref_path'], dtype='float32')
        adv, sr2 = sf.read(final['adv_path'], dtype='float32')
        if sr != sr2 or ref.shape != adv.shape:
            raise ValueError('Audio alignment mismatch')
        outcomes = {}
        for channel in spec['channels']:
            def deliver(x):
                if channel == 'clean':
                    return x
                codec, rate = channel.split('_') if '_' in channel else ('opus', channel[4:])
                return codec_roundtrip(x, sr, codec, int(rate))
            raw = deliver(ref)
            attacked = deliver(adv)
            if a.mode != 'quality':
                import torch
                import torchaudio
                from model_registry import load_model
                if model is None:
                    model = load_model(a.model)
                    # Kimi wraps its torch module inside the inference API.
                    module = model.model.alm if a.model == 'kimi_audio' else model.model
                    module.requires_grad_(False)
                def generate(x):
                    tensor = torchaudio.functional.resample(torch.from_numpy(np.ascontiguousarray(x)).unsqueeze(0), sr, 16000).cuda()
                    with torch.no_grad():
                        return str(model.generate(tensor, prompt=row['prompt']))
                cache_key = (row['audio_hashes']['carrier_ref_path'], row['prompt'], channel)
                if cache_key not in cache:
                    cache[cache_key] = generate(raw)
                text = generate(attacked)
                outcomes[channel] = dict(attacked_output=text, attacked_hit=target_match(row['target'], text),
                    raw_output=cache[cache_key], raw_hit=target_match(row['target'], cache[cache_key]))
            else:
                metrics = {}
                for treatment, x in [('raw', raw), ('attacked', attacked)]:
                    s = snr_db(ref, x)
                    q = dict(snr_db=s if math.isfinite(s) else None, zero_noise=bool(np.array_equal(ref, x)))
                    if row['scenario'] in ['s1', 's2']:
                        from scipy.signal import resample_poly
                        from pesq import pesq
                        from pystoi import stoi
                        g = math.gcd(sr, 16000)
                        rr = resample_poly(ref, 16000//g, sr//g)
                        xx = resample_poly(x, 16000//g, sr//g)
                        try:
                            q.update(pesq_wb=float(pesq(16000, rr, xx, 'wb')), stoi=float(stoi(rr, xx, 16000)), error=None)
                        except Exception as e:
                            q['error'] = repr(e)
                    else:
                        q['speech_metrics'] = 'not applicable to music'
                    metrics[treatment] = q
                outcomes[channel] = metrics
        result = dict(example_id=eid, scenario=row['scenario'], seed=spec['seed'], epsilon=spec['epsilon'],
                      outcomes=outcomes, model=a.model if a.mode != 'quality' else None,
                      excluded_from_optimization=a.mode != 'quality' and a.model not in spec['members'])
        with dest.open('a') as f:
            f.write(json.dumps(result) + '\n'); f.flush()
        print(a.mode, eid, flush=True)


if __name__ == '__main__':
    main()
