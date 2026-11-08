"""Speech-only saved-PCM quality by delivery channel; failures are explicit."""
import argparse, json, math, sys
from pathlib import Path
from math import gcd
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from pesq import pesq
from pystoi import stoi
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from reviewer_controls.common import codec_roundtrip, snr_db, EVAL_CHANNELS

def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,nargs='+',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    done=set()
    if a.output.exists(): done={json.loads(x)['quality_id'] for x in a.output.read_text().splitlines()}
    for manifest in a.manifest:
        for r in map(json.loads,manifest.read_text().splitlines()):
            if r['scenario']!='s1': raise ValueError('Speech S1 only')
            for channel in EVAL_CHANNELS:
                key=str(Path(r['adv_path']).resolve())+':'+channel
                if key in done: continue
                row=dict(quality_id=key,example_id=r['example_id'],method=r['method'],seed=r.get('seed'),channel=channel,reference='raw normalized carrier PCM',reference_path=r['carrier_ref_path'],adv_path=r['adv_path'])
                try:
                    ref,sr=sf.read(r['carrier_ref_path'],dtype='float64');adv,sr2=sf.read(r['adv_path'],dtype='float64')
                    if sr!=sr2 or ref.shape!=adv.shape or ref.ndim!=1: raise ValueError('Sample alignment/rate mismatch')
                    if channel!='clean':
                        codec,rate=channel.split('_') if '_' in channel else ('opus',channel[4:])
                        adv=codec_roundtrip(adv,sr,codec,int(rate))
                    g=gcd(sr,16000); rr=resample_poly(ref,16000//g,sr//g); aa=resample_poly(adv,16000//g,sr//g)
                    snr=snr_db(ref,adv)
                    row.update(snr_db=snr if math.isfinite(snr) else None,zero_noise=bool(np.array_equal(ref,adv)),pesq_wb=float(pesq(16000,rr,aa,'wb')),stoi=float(stoi(rr,aa,16000,extended=False)),metric_sample_rate=16000,error=None)
                except Exception as e: row['error']=repr(e)
                with a.output.open('a') as f: f.write(json.dumps(row)+'\n')
                print(r['example_id'],channel,row.get('error'),flush=True)
if __name__=='__main__':main()
