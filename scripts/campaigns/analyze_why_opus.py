"""Why is Opus the right EoT proxy? Signal-level mechanism analysis.

Analysis of EXISTING adversarial examples. No optimization, no victim model, CPU-only.

Perturbation definition. The baseline is the matched clean reconstruction, i.e. the
carrier pushed through the same EnCodec round-trip that produced the adversarial
audio -- not the raw carrier. Otherwise ordinary codec reconstruction error is
counted as retained adversarial perturbation:

    d    = x_adv          - x_clean_recon
    d_C  = C(x_adv)       - C(x_clean_recon)          (same C applied to both)

Per example and channel:
    cos(d, d_C)                 directional preservation
    ||d_C|| / ||d||             magnitude retention
    ||d_C - d|| / ||d||         relative perturbation distortion
and per example: spectral centroid, fraction of energy below 4 kHz / 400 Hz.

Aggregation follows the ASR table: mean over the 50 pairs within a seed, then
mean +/- SD over seeds {1,7,42}.

Usage:
  python scripts/analyze_why_opus.py --seeds 1,7,42 --neural dac,encodec --device cpu
"""
from __future__ import annotations
import argparse, glob, json, os, sys, time
from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CODEC = ROOT.parent
sys.path.insert(0, str(CODEC))
from demo_ota import apply_opus_compression            # noqa: E402
from aac_channel import apply_aac_compression, apply_mp3_compression  # noqa: E402

SR = 16000
ARMS = [
    ("Opus",                "results_codec_robust"),
    ("MP3",                 "results_codec_robust_mp3"),
    ("AAC-LC",              "results_codec_robust_aac"),
    ("Ensemble{Opus+AAC}",  "results_codec_robust_ensemble"),
    ("DAC",                 "results_codec_robust_dac"),
    ("EnCodec",             "results_codec_robust_encodec"),
]
TRANSPORT = [("opus", 16), ("opus", 24), ("opus", 32), ("opus", 64), ("opus", 128),
             ("mp3", 64), ("aac", 64)]


def _apply(kind, x, br):
    if kind == "opus":
        return apply_opus_compression(x, SR, bitrate_kbps=br)
    if kind == "mp3":
        return apply_mp3_compression(x, SR, bitrate_kbps=br)
    if kind == "aac":
        return apply_aac_compression(x, SR, bitrate_kbps=br)
    raise ValueError(kind)


def _mono(path):
    """Load mono at SR. Adversarial wavs are 24 kHz (EnCodec); carriers are 16 kHz."""
    import torch, torchaudio
    x, sr = sf.read(str(path), dtype="float32")
    if x.ndim > 1:
        x = x.mean(1)
    if sr != SR:
        t = torchaudio.functional.resample(torch.from_numpy(x).unsqueeze(0), sr, SR)
        x = t.squeeze(0).numpy().astype(np.float32)
    return x


def _align(*xs):
    n = min(len(x) for x in xs)
    return [x[:n] for x in xs]


def _band_stats(d):
    D = np.abs(np.fft.rfft(d)) ** 2
    f = np.fft.rfftfreq(len(d), 1.0 / SR)
    tot = D.sum() + 1e-20
    return {"frac_below_400hz": float(D[f <= 400].sum() / tot),
            "frac_below_4khz": float(D[f <= 4000].sum() / tot),
            "centroid_hz": float((f * D).sum() / tot)}


def _mstd(per_seed):
    """mean +/- SD over seeds, from a list of per-seed means."""
    v = [x for x in per_seed if x is not None and np.isfinite(x)]
    if not v:
        return None, None
    return float(np.mean(v)), (float(np.std(v, ddof=1)) if len(v) > 1 else 0.0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default="1,7,42")
    p.add_argument("--limit", type=int, default=0, help="clips per seed; 0 = all")
    p.add_argument("--scenario", default="01_finance_voice_agent")
    p.add_argument("--model", default="qwen25_omni")
    p.add_argument("--neural", default="", help="comma list: dac,encodec")
    p.add_argument("--device", default="cpu")
    p.add_argument("--arms", default="", help="comma list of arm names; empty = all")
    p.add_argument("--out", default="/tmp/why_opus.json")
    args = p.parse_args()

    arms = ARMS
    if args.arms:
        want = [a.strip() for a in args.arms.split(",") if a.strip()]
        unknown = [w for w in want if w not in dict(ARMS)]
        if unknown:
            raise SystemExit(f"unknown arm(s): {unknown}; known: {[a for a, _ in ARMS]}")
        arms = [(n, r) for n, r in ARMS if n in want]

    seeds = [s.strip() for s in args.seeds.split(",") if s.strip()]
    neural = [c.strip() for c in args.neural.split(",") if c.strip()]
    channels = [f"{k}{b}" for k, b in TRANSPORT] + neural

    # EnCodec is always needed: it defines the clean reconstruction baseline.
    import neural_codec_delivery_eval as ncd
    from attacks.latent_codec import EnCodecWrapper
    enc = EnCodecWrapper(device=args.device)
    print(f"[why] EnCodec loaded on {args.device} (clean-reconstruction baseline)", flush=True)
    dac = None
    if "dac" in neural:
        from attacks.latent_codec_dac import DACWrapper
        dac = DACWrapper(device=args.device)
        print(f"[why] DAC loaded on {args.device}", flush=True)

    recon_cache: dict[str, np.ndarray] = {}
    chan_cache: dict[tuple, np.ndarray] = {}
    report = {"channels": channels, "seeds": seeds, "delta": "x_adv - encodec_roundtrip(carrier)",
              "aggregation": "mean over pairs within seed, then mean +/- SD over seeds",
              "arms": {}}
    t0 = time.time()

    for arm_name, root in arms:
        per_seed = {c: {"cos": [], "gain": [], "dist": []} for c in channels}
        per_seed_spec = {k: [] for k in ("frac_below_400hz", "frac_below_4khz", "centroid_hz")}
        n_total = 0

        for sd in seeds:
            bundle = ROOT / root / args.scenario / args.model / f"eps_1.0_seed{sd}"
            summaries = sorted(glob.glob(str(bundle / "audio" / "*_summary.json")))
            if args.limit:
                summaries = summaries[:args.limit]
            if not summaries:
                continue
            acc = {c: {"cos": [], "gain": [], "dist": []} for c in channels}
            spec = {k: [] for k in per_seed_spec}

            for s in summaries:
                j = json.load(open(s))
                wav_path = j.get("wav_path")
                adv_path = s.replace("_summary.json", "_attacked.wav")
                if not (wav_path and os.path.isfile(wav_path) and os.path.isfile(adv_path)):
                    continue
                if wav_path not in recon_cache:
                    recon_cache[wav_path] = np.asarray(
                        ncd.encodec_roundtrip(enc, _mono(wav_path)), dtype=np.float32)
                clean = recon_cache[wav_path]
                adv = _mono(adv_path)
                clean, adv = _align(clean, adv)
                d = adv - clean
                nd = np.linalg.norm(d)
                if nd < 1e-8:
                    continue
                n_total += 1
                for k, v in _band_stats(d).items():
                    spec[k].append(v)

                jobs = [(f"{k}{b}", (lambda x, k=k, b=b: _apply(k, x, b))) for k, b in TRANSPORT]
                if dac is not None:
                    jobs.append(("dac", lambda x: ncd.dac_roundtrip(dac, x)))
                if "encodec" in neural:
                    jobs.append(("encodec", lambda x: ncd.encodec_roundtrip(enc, x)))

                for key, fn in jobs:
                    ck = (wav_path, key)
                    if ck not in chan_cache:
                        chan_cache[ck] = np.asarray(fn(clean), dtype=np.float32)
                    c_clean = chan_cache[ck]
                    c_adv = np.asarray(fn(adv), dtype=np.float32)
                    c_clean, c_adv = _align(c_clean, c_adv)
                    d_c = c_adv - c_clean
                    d2, d_c2 = _align(d, d_c)
                    n2, ndc = np.linalg.norm(d2), np.linalg.norm(d_c2)
                    if n2 < 1e-8 or ndc < 1e-8:
                        continue
                    acc[key]["cos"].append(float(np.dot(d2, d_c2) / (n2 * ndc)))
                    acc[key]["gain"].append(float(ndc / n2))
                    acc[key]["dist"].append(float(np.linalg.norm(d_c2 - d2) / n2))

            for c in channels:
                for m in ("cos", "gain", "dist"):
                    per_seed[c][m].append(float(np.mean(acc[c][m])) if acc[c][m] else None)
            for k in spec:
                per_seed_spec[k].append(float(np.mean(spec[k])) if spec[k] else None)
            print(f"[why] {arm_name:20s} seed {sd:>2s} done "
                  f"({len(summaries)} clips, {time.time()-t0:.0f}s elapsed)", flush=True)

        report["arms"][arm_name] = {
            "n_examples": n_total,
            **{m: {c: dict(zip(("mean", "sd"), _mstd(per_seed[c][m]))) for c in channels}
               for m in ("cos", "gain", "dist")},
            "spectrum": {k: dict(zip(("mean", "sd"), _mstd(v))) for k, v in per_seed_spec.items()},
        }
        sp = report["arms"][arm_name]["spectrum"]
        print(f"[why] == {arm_name:20s} n={n_total:4d}  "
              f"<400Hz={sp['frac_below_400hz']['mean']:.3f}  "
              f"<4kHz={sp['frac_below_4khz']['mean']:.3f}  "
              f"centroid={sp['centroid_hz']['mean']:7.1f} Hz", flush=True)
        json.dump(report, open(args.out, "w"), indent=2)   # checkpoint per arm

    json.dump(report, open(args.out, "w"), indent=2)
    print(f"[why] wrote {args.out}  ({time.time()-t0:.0f}s total)", flush=True)


if __name__ == "__main__":
    main()
