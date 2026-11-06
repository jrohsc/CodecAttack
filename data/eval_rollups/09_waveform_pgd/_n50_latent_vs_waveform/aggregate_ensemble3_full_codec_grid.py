#!/usr/bin/env python3
"""Validate and summarize all paper delivery codecs on the saved 224-target ensemble."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / 'results'
PREFIX = 'benchmark_ensemble3_opuseot_transcribe_eps10.0_jazz_1_bestpostcodec_steps600'
CHANNELS = ('clean','opus16','opus24','opus32','opus64','opus128','opus192',
            'mp3_64','mp3_96','mp3_128','mp3_192',
            'aac_64','aac_96','aac_128','aac_192','dac','encodec')
MODELS = (
    ('qwen2_MEMBER','Qwen2-Audio','member','qwen2_audio'),
    ('omni7b_MEMBER','Qwen2.5-Omni-7B','member','qwen25_omni'),
    ('af3_MEMBER','Audio Flamingo 3','member','audio_flamingo'),
    ('omni3b_HELDOUT','Qwen2.5-Omni-3B','held out','qwen25_omni_3b'),
    ('kimi_HELDOUT_100tok','Kimi-Audio-7B-Instruct','held out','kimi_audio'),
    ('phi4_HELDOUT','Phi-4-multimodal','held out','phi4'),
)

def main():
    result={'attack_prefix':PREFIX,'n':224,'metric':'case-insensitive strict target substring',
            'channels':list(CHANNELS),'models':{}}
    lines=['# Full paper codec grid on the saved 224-target ensemble attack','',
           'Strict target-substring ASR (%). The attack uses zero initialization, epsilon 10, seed 1, 600 steps, and Opus EoT. Neural delivery means DAC or EnCodec resynthesis of the saved waveform.','',
           '| Role | Victim | '+' | '.join(CHANNELS)+' |',
           '|---|---|'+'|'.join('---:' for _ in CHANNELS)+'|']
    all_clips=set()
    for out,label,role,model in MODELS:
        counts={ch:0 for ch in CHANNELS}
        clips=set()
        for sh in range(4):
            bundle=ROOT/f'{PREFIX}_shard{sh}of4'
            config=json.loads((bundle/'config.json').read_text())
            assert config['attack_steps']==600 and config['eps']==10.0
            assert config['ensemble_channel_mode']=='multi_bitrate'
            assert config['rand_init'] is False
            wavs={p.stem for p in (bundle/'audio').glob('*.wav')}
            assert len(wavs)==56,(bundle,len(wavs))
            records={p.parent.name:json.loads(p.read_text()) for p in (bundle/out).glob('*/record.json')}
            assert set(records)==wavs,(bundle,out,len(records),len(wavs))
            clips.update(records)
            for clip,rec in records.items():
                assert rec['target_model']==model and rec['target_text'],(bundle,out,clip)
                for ch in CHANNELS:
                    assert isinstance(rec.get(ch),dict) and isinstance(rec[ch].get('exact_match'),bool),(bundle,out,clip,ch)
                    counts[ch]+=rec[ch]['exact_match']
        assert len(clips)==224,(out,len(clips))
        if not all_clips: all_clips=clips
        else: assert clips==all_clips,out
        pct={ch:round(100*counts[ch]/224,1) for ch in CHANNELS}
        result['models'][out]={'label':label,'role':role,'target_model':model,'counts':counts,'percentages':pct}
        lines.append('| '+ ' | '.join((role,label,*[f'{pct[ch]:.1f}' for ch in CHANNELS]))+' |')
    jp=ROOT/f'{PREFIX}_FULL_CODEC_GRID_AGGREGATED.json'
    mp=ROOT/f'{PREFIX}_FULL_CODEC_GRID_AGGREGATED.md'
    jp.write_text(json.dumps(result,indent=2)+'\n')
    mp.write_text('\n'.join(lines)+'\n')
    print(jp);print(mp)

if __name__=='__main__': main()
