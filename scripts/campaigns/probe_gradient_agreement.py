"""
Do the surrogates' INPUT-SPACE GRADIENTS agree with each other at all?

Every proposed fix for the transfer null -- gradient averaging, sign voting,
common-subspace (SVD) updates, universal perturbations, momentum/variance
reduction -- is a way of COMBINING per-model gradients. All of them presuppose
that the models' gradients point in related directions. That premise has never
been measured here. probe_encoder_agreement.py measured whether the attacked
audio lands in the same place in each model's *representation*; this measures
something upstream of that and much more actionable: whether dL/dx, the thing an
ensemble attack actually averages, is shared.

For each (carrier, target) pair we evaluate every surrogate's gradient at the
SAME three input points and compare them:

    clean  the unattacked carrier -- where a fresh ensemble attack starts
    mid    halfway between clean and adv (a point on the segment, NOT a
           replayed PGD iterate; labelled honestly as an interpolant)
    adv    the qwen2_audio-converged adversarial wav from the transcribe-prompt
           campaign -- where the source attack ended

Metrics, per model pair per point:
    cos_wave    cos(g_i, g_j) on the raw 16 kHz waveform gradient
                -> can you average gradients at all
    sign_agree  mean[ sign(g_i) == sign(g_j) ] over the JOINT SUPPORT (samples
                where both gradients are non-zero), chance = 0.5; `supp` reports
                what fraction of the clip that support is
                -> is sign voting better than magnitude averaging
    cos_band    correlation of mean-centred LOG per-frequency-bin gradient
                energy -> is there a shared spectral SHAPE even when the
                waveform-domain direction does not agree at all
plus, over all models jointly:
    svd_top1    energy fraction of the leading principal component of the
                stacked M x T normalised gradient matrix; isotropic = 1/M
                -> is there a common subspace to project onto

Three controls, because a bare cosine in 160k dimensions means nothing:
    noise    same model, same input, queried TWICE. bf16 forward/backward is not
             bit-exact, so this is the measurement ceiling. A cross-model cosine
             must be read against it, never against 1.0.
    random   two Gaussian vectors of the same dimension -- the analytic floor,
             computed empirically so the table carries its own zero.
    carrier  same model, two DIFFERENT carriers -- the ceiling for a universal
             (carrier-agnostic) perturbation. If a model does not agree with
             itself across carriers, one perturbation cannot serve many.
And one calibration anchor: qwen25_omni vs qwen25_omni_3b is the ONLY pair with
a non-zero measured transfer rate (1.1%). Whatever the metrics read on that pair
is what "barely enough to transfer" looks like in these units.

Each model needs its own conda env, so collection is one process per model
(gradients dumped to .npy) and --summarize does the cross-model maths.

Usage:
    python probe_gradient_agreement.py --prepare --pairs 12
    python probe_gradient_agreement.py --model qwen2_audio      # repeat per model
    python probe_gradient_agreement.py --summarize
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from config import CONDA_ENVS, PROJECT_ROOT  # noqa: E402
from model_registry import DIFFERENTIABLE, ENCODER_FAMILY  # noqa: E402

TRANSCRIBE_PROMPT = "What does the person say in this audio?"
SR = 16000
ADV_BUNDLE = HERE / "0_all_combined/results_transcribe_prompt/01_finance_voice_agent/qwen2_audio/eps_1.0/audio"
DEFAULT_OUT = HERE / "results/gradient_agreement"
POINTS = ["clean", "mid", "adv"]
MODELS = ["qwen2_audio", "qwen25_omni", "qwen25_omni_3b", "audio_flamingo", "phi4", "gemma3n"]


# --------------------------------------------------------------------------
# Phase 1: build the shared input points
# --------------------------------------------------------------------------

def _load_16k(path, duration=None):
    """Mirror music_carrier.load_music() but at 16 kHz, the models' input rate."""
    import librosa
    wav, _ = librosa.load(str(path), sr=SR, mono=True, duration=duration)
    if duration is not None:
        n = int(round(duration * SR))
        wav = np.pad(wav, (0, max(0, n - len(wav))))[:n]
    return wav.astype(np.float32)


def prepare(outdir: Path, n_pairs: int):
    """Write manifest.json + one .npy per (pair, point), all at 16 kHz."""
    summaries = sorted(ADV_BUNDLE.glob("*_summary.json"))
    if not summaries:
        sys.exit(f"No attack summaries under {ADV_BUNDLE}")

    ptdir = outdir / "points"
    ptdir.mkdir(parents=True, exist_ok=True)
    manifest = []

    for sp in summaries:
        if len(manifest) >= n_pairs:
            break
        s = json.loads(sp.read_text())
        adv_path = sp.with_name(sp.name.replace("_summary.json", "_attacked.wav"))
        if not adv_path.exists():
            continue
        dur = float(s["carrier_duration_s"])

        # normalize=True in load_music() is applied to the CARRIER only; the adv
        # wav was written post-attack and must be taken as-is, so the delta we
        # interpolate along is the real perturbation.
        clean = _load_16k(s["wav_path"], duration=dur)
        peak = np.abs(clean).max()
        if peak > 0:
            clean = clean / (peak + 1e-8) * 0.95
        adv = _load_16k(adv_path, duration=dur)
        n = min(len(clean), len(adv))
        clean, adv = clean[:n], adv[:n]

        stem = sp.name.replace("_eps1_summary.json", "")
        for name, x in (("clean", clean), ("mid", clean + 0.5 * (adv - clean)), ("adv", adv)):
            np.save(ptdir / f"{stem}__{name}.npy", x)
        manifest.append({
            "stem": stem,
            "target_text": s["target_text"],
            "carrier_wav": s["wav_path"],
            "duration_s": dur,
            "n_samples": int(n),
            "source_success": bool(s.get("success")),
        })

    (outdir / "manifest.json").write_text(json.dumps(
        {"prompt": TRANSCRIBE_PROMPT, "sr": SR, "points": POINTS, "pairs": manifest}, indent=2))
    print(f"prepared {len(manifest)} pairs x {len(POINTS)} points -> {ptdir}")
    lens = {m["n_samples"] for m in manifest}
    print(f"  sample lengths present: {sorted(lens)}")


# --------------------------------------------------------------------------
# Phase 2: collect one model's gradients (runs in that model's conda env)
# --------------------------------------------------------------------------

def _spawn_worker(model_name: str):
    """Copy of ensemble_attack._get_persistent_worker's launch, minus the state."""
    conda_env = CONDA_ENVS[model_name]
    conda_exe = os.environ.get("CONDA_EXE", "")
    if conda_exe:
        conda_base = os.path.dirname(os.path.dirname(conda_exe))
    else:
        prefix = os.path.abspath(sys.prefix)
        conda_base = (os.path.dirname(os.path.dirname(prefix))
                      if os.path.basename(os.path.dirname(prefix)) == "envs" else prefix)
    activate = os.path.join(conda_base, "etc", "profile.d", "conda.sh")
    script = os.path.join(PROJECT_ROOT, "gradient_worker_persistent.py")
    # Absolute interpreter path: `conda activate` loses the PATH race against a
    # caller that prepended another env's bin, and an out-of-env worker fails in
    # a way that looks like a missing gradient rather than a broken env.
    env_python = os.path.join(conda_base, "envs", conda_env, "bin", "python")
    cmd = (f'source "{activate}" && conda activate {conda_env} && '
           f'"{env_python}" -u "{script}" --model {model_name}')

    errdir = Path(PROJECT_ROOT) / "logs" / "workers"
    errdir.mkdir(parents=True, exist_ok=True)
    err_path = errdir / f"gradprobe_{model_name}_{os.getpid()}.err"
    err_fh = open(err_path, "w")
    print(f"[{model_name}] launching worker (env={conda_env}, stderr={err_path})")
    proc = subprocess.Popen(["bash", "-c", cmd], stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=err_fh, text=True,
                            bufsize=1, env={**os.environ, "HF_HUB_OFFLINE": "1"})
    for _ in range(400):
        line = proc.stdout.readline()
        if not line:
            break
        if line.strip() == "READY":
            return proc, err_path
        print(f"[{model_name}] worker stdout before READY: {line.strip()!r}")
    proc.kill()
    sys.exit(f"[{model_name}] worker never became READY; see {err_path}")


def collect(outdir: Path, model_name: str, repeat_control: bool):
    man = json.loads((outdir / "manifest.json").read_text())
    gdir = outdir / "grads" / model_name
    gdir.mkdir(parents=True, exist_ok=True)
    proc, err_path = _spawn_worker(model_name)
    tmpdir = Path(tempfile.mkdtemp(prefix=f"gradprobe_{model_name}_"))
    losses = {}

    def ask(x: np.ndarray, target: str, tag: str):
        req = tmpdir / f"{tag}.pt"
        torch.save({"audio_16k": torch.from_numpy(x).float().unsqueeze(0),
                    "target_text": target, "prompt": TRANSCRIBE_PROMPT}, req)
        proc.stdin.write(str(req) + "\n")
        proc.stdin.flush()
        reply = proc.stdout.readline().strip()
        if not reply.startswith("OK"):
            print(f"[{model_name}] {tag}: {reply}")
            req.unlink(missing_ok=True)
            return None, None
        out = torch.load(str(req) + ".out", map_location="cpu", weights_only=True)
        g = out["gradient"].squeeze(0).float().numpy()
        req.unlink(missing_ok=True)
        Path(str(req) + ".out").unlink(missing_ok=True)
        return g, float(out["loss"])

    try:
        for pair in man["pairs"]:
            stem, tgt = pair["stem"], pair["target_text"]
            for point in POINTS:
                x = np.load(outdir / "points" / f"{stem}__{point}.npy")
                g, loss = ask(x, tgt, f"{stem}__{point}")
                if g is None:
                    continue
                np.save(gdir / f"{stem}__{point}.npy", g)
                losses[f"{stem}__{point}"] = loss
                print(f"[{model_name}] {stem[:44]:44s} {point:5s} loss={loss:.4f}")
            if repeat_control:
                # Same input, second query: bf16 fwd/bwd is not bit-exact, and a
                # cross-model cosine is meaningless without knowing this ceiling.
                x = np.load(outdir / "points" / f"{stem}__clean.npy")
                g, loss = ask(x, tgt, f"{stem}__clean_rep")
                if g is not None:
                    np.save(gdir / f"{stem}__clean_rep.npy", g)
                    losses[f"{stem}__clean_rep"] = loss
    finally:
        try:
            proc.stdin.write("QUIT\n")
            proc.stdin.flush()
            proc.wait(timeout=60)
        except Exception:
            proc.kill()

    (gdir / "losses.json").write_text(json.dumps(losses, indent=2))
    print(f"[{model_name}] wrote {len(list(gdir.glob('*.npy')))} gradients -> {gdir}")


# --------------------------------------------------------------------------
# Phase 3: cross-model statistics
# --------------------------------------------------------------------------

def _cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def _band_energy(g, n_fft=512, hop=160):
    """Mean-centred LOG per-frequency-bin gradient energy (257-vector).

    The obvious version -- cosine of the raw |STFT|-summed-over-time energies --
    is useless: energy vectors are non-negative and roughly flat, so two
    independent Gaussian vectors score 0.9996 on it and the metric has no
    dynamic range. Taking the log and removing the mean makes it a correlation
    of spectral SHAPE about a flat spectrum. Its empirical floor is ~0.20 rather
    than 0 (the STFT window leaves DC/Nyquist bins systematically different in
    any signal), which is why the random control is printed beside every cell.
    """
    import librosa
    S = np.abs(librosa.stft(g.astype(np.float32), n_fft=n_fft, hop_length=hop))
    v = np.log10(S.sum(axis=1) + 1e-20)
    return v - v.mean()


def _sign_agree(a, b):
    """Sign agreement over the JOINT SUPPORT only, and the support size.

    load_music() zero-pads the carrier out to the requested duration, so ~59% of
    a 10 s clean point is digital silence where every model returns an
    identically-zero gradient. Counting sign(0)==sign(0) as agreement scored
    0.80 there while the cosine was 0.006 -- an incoherent pair of numbers that
    was pure padding. Only samples where BOTH gradients are non-zero carry
    information about whether sign voting would work.
    """
    m = (a != 0) & (b != 0)
    k = int(m.sum())
    if k == 0:
        return float("nan"), 0.0
    return float(np.mean(np.sign(a[m]) == np.sign(b[m]))), k / len(a)


def summarize(outdir: Path):
    man = json.loads((outdir / "manifest.json").read_text())
    pairs = man["pairs"]
    present = [m for m in MODELS if (outdir / "grads" / m).is_dir()
               and any((outdir / "grads" / m).glob("*.npy"))]
    if len(present) < 2:
        sys.exit(f"Need >=2 models with gradients; found {present}")
    print(f"models: {present}\npairs:  {len(pairs)}\n")

    def load(m, stem, point):
        p = outdir / "grads" / m / f"{stem}__{point}.npy"
        return np.load(p) if p.exists() else None

    rows = []          # cross-model, per point
    for point in POINTS:
        for i, mi in enumerate(present):
            for mj in present[i + 1:]:
                cw, sa, cb, supp = [], [], [], []
                for pr in pairs:
                    gi, gj = load(mi, pr["stem"], point), load(mj, pr["stem"], point)
                    if gi is None or gj is None:
                        continue
                    n = min(len(gi), len(gj))
                    cw.append(_cos(gi[:n], gj[:n]))
                    s_, sup_ = _sign_agree(gi[:n], gj[:n])
                    sa.append(s_); supp.append(sup_)
                    cb.append(_cos(_band_energy(gi[:n]), _band_energy(gj[:n])))
                if cw:
                    rows.append({"point": point, "a": mi, "b": mj, "n": len(cw),
                                 "cos_wave": float(np.mean(cw)), "cos_wave_sd": float(np.std(cw)),
                                 "sign_agree": float(np.mean(sa)),
                                 "support": float(np.mean(supp)),
                                 "cos_band": float(np.mean(cb)), "cos_band_sd": float(np.std(cb)),
                                 "same_family": ENCODER_FAMILY[mi] == ENCODER_FAMILY[mj]})

    # Controls -------------------------------------------------------------
    ctrl = {}
    rng = np.random.default_rng(0)
    T = pairs[0]["n_samples"]
    r1, r2 = rng.standard_normal(T), rng.standard_normal(T)
    r1, r2 = r1.astype(np.float32), r2.astype(np.float32)
    ctrl["random"] = {"cos_wave": _cos(r1, r2), "sign_agree": _sign_agree(r1, r2)[0],
                      "cos_band": _cos(_band_energy(r1), _band_energy(r2))}
    for m in present:
        cw, sa, cb = [], [], []
        for pr in pairs:
            g1, g2 = load(m, pr["stem"], "clean"), load(m, pr["stem"], "clean_rep")
            if g1 is None or g2 is None:
                continue
            cw.append(_cos(g1, g2)); sa.append(_sign_agree(g1, g2)[0])
            cb.append(_cos(_band_energy(g1), _band_energy(g2)))
        if cw:
            ctrl.setdefault("noise", {})[m] = {
                "cos_wave": float(np.mean(cw)), "sign_agree": float(np.mean(sa)),
                "cos_band": float(np.mean(cb)), "n": len(cw)}
    for m in present:
        # Same model, clean vs converged-adv: does the direction the attack
        # starts in survive to where it ends? If not, agreement measured at the
        # clean point says nothing about the point an ensemble actually reaches.
        cw, cb = [], []
        for pr in pairs:
            g1, g2 = load(m, pr["stem"], "clean"), load(m, pr["stem"], "adv")
            if g1 is None or g2 is None:
                continue
            cw.append(_cos(g1, g2))
            cb.append(_cos(_band_energy(g1), _band_energy(g2)))
        if cw:
            ctrl.setdefault("traj", {})[m] = {
                "cos_wave": float(np.mean(cw)), "cos_band": float(np.mean(cb)), "n": len(cw)}
    for m in present:
        cw, cb = [], []
        for a in range(len(pairs)):
            for b in range(a + 1, len(pairs)):
                ga, gb = load(m, pairs[a]["stem"], "clean"), load(m, pairs[b]["stem"], "clean")
                if ga is None or gb is None:
                    continue
                n = min(len(ga), len(gb))
                cw.append(_cos(ga[:n], gb[:n]))
                cb.append(_cos(_band_energy(ga[:n]), _band_energy(gb[:n])))
        if cw:
            ctrl.setdefault("carrier", {})[m] = {
                "cos_wave": float(np.mean(cw)), "cos_band": float(np.mean(cb)), "n": len(cw)}

    # Common subspace ------------------------------------------------------
    svd = {}
    for point in POINTS:
        fracs = []
        for pr in pairs:
            G = []
            for m in present:
                g = load(m, pr["stem"], point)
                if g is not None:
                    nrm = np.linalg.norm(g)
                    if nrm > 0:
                        G.append(g[:pr["n_samples"]] / nrm)
            if len(G) < 2:
                continue
            s = np.linalg.svd(np.stack(G), compute_uv=False)
            fracs.append(float((s[0] ** 2) / (s ** 2).sum()))
        if fracs:
            svd[point] = {"top1_energy": float(np.mean(fracs)),
                          "isotropic": 1.0 / len(present), "n": len(fracs)}

    out = {"models": present, "n_pairs": len(pairs), "cross": rows,
           "controls": ctrl, "svd": svd}
    (outdir / "agreement.json").write_text(json.dumps(out, indent=2))
    _print_tables(out)


def _print_tables(out):
    w = 34
    print("=" * 96)
    print("CONTROLS  (what the numbers mean before you read any cross-model cell)")
    print("=" * 96)
    print(f"{'control':{w}s} {'cos_wave':>10s} {'sign_agree':>11s} {'cos_band':>10s}  note")
    c = out["controls"]["random"]
    print(f"{'random Gaussian (floor)':{w}s} {c['cos_wave']:10.4f} {c['sign_agree']:11.4f} "
          f"{c['cos_band']:10.4f}  analytic zero")
    for m, v in out["controls"].get("noise", {}).items():
        print(f"{'noise: ' + m + ' vs itself':{w}s} {v['cos_wave']:10.4f} {v['sign_agree']:11.4f} "
              f"{v['cos_band']:10.4f}  measurement ceiling")
    for m, v in out["controls"].get("carrier", {}).items():
        print(f"{'carrier: ' + m + ' across clips':{w}s} {v['cos_wave']:10.4f} {'-':>11s} "
              f"{v['cos_band']:10.4f}  universal-perturbation ceiling")
    for m, v in out["controls"].get("traj", {}).items():
        print(f"{'traj: ' + m + ' clean vs adv':{w}s} {v['cos_wave']:10.4f} {'-':>11s} "
              f"{v['cos_band']:10.4f}  does the start direction survive")

    for point in POINTS:
        rows = [r for r in out["cross"] if r["point"] == point]
        if not rows:
            continue
        print()
        print("=" * 96)
        print(f"CROSS-MODEL GRADIENT AGREEMENT @ {point}")
        print("=" * 96)
        print(f"{'pair':{w}s} {'cos_wave':>10s} {'±sd':>7s} {'sign_agree':>11s} "
              f"{'cos_band':>10s} {'±sd':>7s} {'supp':>6s}  fam")
        for r in sorted(rows, key=lambda r: -r["cos_band"]):
            fam = "same" if r["same_family"] else "cross"
            name = f"{r['a']} / {r['b']}"
            print(f"{name:{w}.{w}s} {r['cos_wave']:10.4f} {r['cos_wave_sd']:7.4f} "
                  f"{r['sign_agree']:11.4f} {r['cos_band']:10.4f} {r['cos_band_sd']:7.4f} "
                  f"{r['support']:6.3f}  {fam}")
    print()
    print("=" * 96)
    print("COMMON SUBSPACE  (top-1 PC energy of the stacked normalised gradients)")
    print("=" * 96)
    for point, v in out["svd"].items():
        print(f"  {point:6s} top1={v['top1_energy']:.4f}  isotropic={v['isotropic']:.4f}  "
              f"n={v['n']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", default=str(DEFAULT_OUT))
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--pairs", type=int, default=12)
    ap.add_argument("--model", help="collect gradients for this model")
    ap.add_argument("--no-repeat-control", action="store_true")
    ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args()
    outdir = Path(a.outdir)

    if a.prepare:
        prepare(outdir, a.pairs)
    if a.model:
        if a.model not in DIFFERENTIABLE:
            sys.exit(f"{a.model} is not differentiable; surrogates: {sorted(DIFFERENTIABLE)}")
        collect(outdir, a.model, repeat_control=not a.no_repeat_control)
    if a.summarize:
        summarize(outdir)
    if not (a.prepare or a.model or a.summarize):
        ap.print_help()


if __name__ == "__main__":
    main()
