"""
Ensemble adversarial attack on multiple Audio LLMs via gradient stitching.

Optimizes a single latent-space perturbation against multiple models
simultaneously by accumulating per-model gradients at the audio_16k level,
then backpropagating through the shared EnCodec decoder.

Gradient flow:
    delta (learnable) -> z_adv = z_original + delta
    -> EnCodec.decode(z_adv) -> audio_24kHz -> resample -> audio_16kHz
                                                             |
                                                     (detach + clone)
                                              ┌── Model A: compute_loss -> grad_A
                                              ├── Model B: compute_loss -> grad_B
                                              └── Model C (subprocess): grad_C via file
                                                             |
                                        audio_16k.backward(weighted_sum_of_grads)
                                                             |
                                                     delta.grad updated

Models in the same conda env run in-process; models requiring different envs
(e.g. Audio Flamingo 3 needs transformers 5.x) run via gradient_worker.py
in a subprocess.
"""

import os
import sys
import json
import time
import subprocess
import tempfile
from typing import Dict, List, Optional, Tuple

import torch
import torchaudio
import numpy as np

from config import (
    WHISPER_INJECT_ROOT, MODEL_PATHS, TARGET_MODEL,
    ENCODEC_BANDWIDTH, ENCODEC_SAMPLE_RATE, TARGET_SAMPLE_RATE,
    LATENT_EPS, LATENT_ALPHA, ATTACK_STEPS, CHECK_EVERY,
    PERCEPTUAL_WEIGHT, compute_wer,
    ENSEMBLE_MODELS, ENSEMBLE_WEIGHTS, ENSEMBLE_IN_PROCESS,
    ENSEMBLE_SUBPROCESS, ENSEMBLE_WARMUP_STEPS, ENSEMBLE_GRAD_NORMALIZE,
    CONDA_ENVS, PROJECT_ROOT, SCRIPTS_DIR,
)
from music_carrier import mel_distance, multi_scale_mel_distance
from latent_attack import AttackResult, LatentCodecAttacker, UNTARGETED_WER_THRESHOLD

sys.path.insert(0, WHISPER_INJECT_ROOT)


def _load_model(model_name: str, device: str, dtype: torch.dtype):
    """Load a target model by name. Returns the model wrapper instance.

    Delegates to model_registry so the three loaders (here, gradient_worker.py,
    gradient_worker_persistent.py) cannot drift apart as the pool grows.
    """
    from model_registry import load_model
    return load_model(model_name, device=device, dtype=dtype)


def persistent_worker_command(model_name, conda_base, script):
    """Allow explicit venv interpreters without changing legacy conda routing."""
    override = os.environ.get(f"CODECATTACK_PYTHON_{model_name.upper()}")
    if override:
        if not os.path.isabs(override) or not os.path.isfile(override) or not os.access(override, os.X_OK):
            raise ValueError(f"Invalid worker interpreter for {model_name}: {override}")
        return [override, "-u", script, "--model", model_name]
    env_name = CONDA_ENVS[model_name]
    activate = os.path.join(conda_base, "etc", "profile.d", "conda.sh")
    python = os.path.join(conda_base, "envs", env_name, "bin", "python")
    return ["bash", "-c", f'source "{activate}" && conda activate {env_name} && "{python}" -u "{script}" --model {model_name}']


class EnsembleLatentAttacker:
    """
    Ensemble adversarial attack using gradient stitching across multiple
    Audio LLMs. Produces a single perturbation that fools multiple models.

    Compatible with LatentCodecAttacker interface: returns AttackResult,
    supports test_opus_robustness(), save_adversarial_wav(), etc.
    """

    def __init__(
        self,
        ensemble_models: List[str] = None,
        ensemble_weights: Dict[str, float] = None,
        in_process_models: List[str] = None,
        subprocess_models: List[str] = None,
        primary_model: str = TARGET_MODEL,
        encodec_bandwidth: float = ENCODEC_BANDWIDTH,
        eps: float = LATENT_EPS,
        alpha: float = LATENT_ALPHA,
        perceptual_weight: float = PERCEPTUAL_WEIGHT,
        warmup_steps: int = ENSEMBLE_WARMUP_STEPS,
        grad_normalize: bool = ENSEMBLE_GRAD_NORMALIZE,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        verbose: bool = True,
        rand_init: bool = False,
        seed: int = 1,
        rand_init_scale: float = 0.01,
        channel_mode: str = "none",
        multi_bitrate_kbps: List[int] = None,
        mp3_bitrates_kbps: List[int] = None,
        channel_warmup_ratio: float = 0.3,
        model_eot: bool = False,
        strict_workers: bool = False,
        grad_cache: bool = False,
        cache_block: int = 1,
        cache_ttl_mult: float = 2.0,
        cpu_offload: bool = False,
        best_after_warmup: bool = False,
        diagnostic_interval: int = 0,
    ):
        # best_after_warmup: only steps where EVERY member is in the loss and the
        # codec EoT proxy (if any) is on may become best_delta. Off by default so
        # existing bundles reproduce. Without it, the primary-only warmup loss
        # (~0.01) beats post-warmup ensemble totals, so the saved wav is a
        # primary-only, un-hardened perturbation (E2 latched on 191/215 clips).
        self.diagnostic_interval = int(diagnostic_interval)
        self.best_after_warmup = bool(best_after_warmup)
        # Multi-bitrate Opus EoT. The ensemble attacker was originally
        # clean-channel only (its test_opus_robustness is eval-time), which is
        # why the transcribe-mode ensemble run needed eps=10.0 to survive Opus.
        # channel_mode="multi_bitrate" routes the decoded audio through the same
        # DifferentiableOpusProxy (STE/BPDA) that robust_latent_attack.py uses,
        # so an ensemble arm can be run under the paper's codec-hardened
        # conditions. Add-only: channel_mode="none" keeps the original path.
        self.channel_mode = channel_mode
        self.channel_warmup_ratio = float(channel_warmup_ratio)
        self._multi_bitrate_kbps = multi_bitrate_kbps or [16, 24, 32, 64, 128]
        self.opus_proxy = None
        self.mp3_proxy = None
        if channel_mode == "multi_bitrate":
            from channel_augmentation import DifferentiableOpusProxy
            self.opus_proxy = DifferentiableOpusProxy(
                sample_rate=ENCODEC_SAMPLE_RATE,
                bitrates_kbps=self._multi_bitrate_kbps,
            )
        elif channel_mode == "mp3_multibitrate":
            # Reuse the same MP3 STE/BPDA proxy and bitrate grid as the
            # single-victim codec-EoT ablation.  The forward pass is real
            # ffmpeg MP3 while the backward pass is the identity estimator.
            from aac_channel import AACCodecSTE
            self._mp3_bitrates_kbps = mp3_bitrates_kbps or [32, 64, 96, 128]
            self.mp3_proxy = AACCodecSTE(
                sample_rate=ENCODEC_SAMPLE_RATE,
                mp3_bitrates=self._mp3_bitrates_kbps,
                mp3_only=True,
            ).to(device)
        elif channel_mode != "none":
            raise ValueError(f"unsupported channel_mode: {channel_mode}")

        self.eps = eps
        self.alpha = alpha
        self.perceptual_weight = perceptual_weight
        self.rand_init = rand_init
        self.seed = seed
        self.rand_init_scale = rand_init_scale
        self.warmup_steps = warmup_steps
        self.grad_normalize = grad_normalize
        self.device = device
        self.dtype = dtype
        self.verbose = verbose
        self.primary_model = primary_model

        # Resolve model lists
        self.ensemble_models = ensemble_models or ENSEMBLE_MODELS
        self.ensemble_weights = ensemble_weights or ENSEMBLE_WEIGHTS
        self.in_process_models = in_process_models if in_process_models is not None else ENSEMBLE_IN_PROCESS
        self.subprocess_models = subprocess_models if subprocess_models is not None else ENSEMBLE_SUBPROCESS

        # EoT over the victim axis: sample one model per step instead of
        # averaging all of them. Same expected objective as the mean (uniform
        # weights), ~1/N the per-step cost. Default OFF so every existing run
        # keeps its published gradient-averaging behaviour.
        import random as _random
        self.model_eot = bool(model_eot)
        self.strict_workers = bool(strict_workers)

        # --- Stale-gradient cache (scaling the surrogate pool) ---------------
        # The stitching point is audio_16k, so a surrogate's entire contribution
        # to a step is one [1, T] vector. Nothing forces all N of them to be
        # recomputed on the SAME step. grad_cache keeps the most recent gradient
        # per surrogate and refreshes ONE of them per step (round-robin, in
        # blocks of cache_block), combining the full cache every step:
        #
        #   combined = sum_m w_m * g_m(t - k_m),   k_m <= cache_block * N
        #
        # Cost per step is 1 fwd+bwd regardless of N -- the same as --model-eot --
        # but the STEP DIRECTION is the full ensemble rather than a single
        # sampled member, so the optimizer is not chasing one victim at a time.
        # This is what makes N > 3 affordable. Its validity rests on the
        # gradients being temporally autocorrelated over k steps, which the run
        # measures for itself (history["grad_staleness"]).
        #
        # Default OFF: every existing bundle reproduces byte-for-byte.
        self.grad_cache = bool(grad_cache) or os.environ.get("ENSEMBLE_GRAD_CACHE", "0") == "1"
        self.cache_block = max(1, int(cache_block))
        self.cache_ttl_mult = float(cache_ttl_mult)
        # cpu_offload parks the idle in-process surrogates in host RAM between
        # their blocks, so resident VRAM is one model + EnCodec regardless of N.
        # A 7B bf16 model is ~14 GB over PCIe (~1-2 s), which is amortized away
        # by cache_block; reloading from disk (~46 s for AF3) would not be.
        self.cpu_offload = bool(cpu_offload) or os.environ.get("ENSEMBLE_CPU_OFFLOAD", "0") == "1"
        if self.cpu_offload and not self.grad_cache:
            raise ValueError("cpu_offload requires grad_cache=True: without a "
                             "cache every model is needed every step, so parking "
                             "them on CPU would thrash the bus.")
        self._eot_rng = _random.Random(seed)

        # Filter to only requested models
        self.in_process_models = [m for m in self.in_process_models if m in self.ensemble_models]
        self.subprocess_models = [m for m in self.subprocess_models if m in self.ensemble_models]

        # Auto-route anything the two static lists do not cover. ENSEMBLE_IN_PROCESS
        # / ENSEMBLE_SUBPROCESS predate the 6-model pool and name only 3 models, so
        # a surrogate that is in ensemble_models but in NEITHER list used to be
        # dropped here without a word -- the run would log a healthy ensemble,
        # rescale the surviving weights, and quietly optimize against fewer victims
        # than requested. That is indistinguishable in the results from "adding
        # this surrogate didn't help", which is exactly the claim the pool-scaling
        # campaign is trying to measure. Route by conda env, then hard-fail on
        # anything still unaccounted for.
        _routed = set(self.in_process_models) | set(self.subprocess_models)
        _unrouted = [m for m in self.ensemble_models if m not in _routed]
        if _unrouted:
            _here = os.path.basename(sys.prefix)
            for m in _unrouted:
                if CONDA_ENVS.get(m, _here) == _here:
                    self.in_process_models.append(m)
                else:
                    self.subprocess_models.append(m)

        # A single-model cell whose model is hard-listed in ENSEMBLE_SUBPROCESS
        # (audio_flamingo, phi4) leaves in_process_models EMPTY, and the ctor dies
        # at "At least one in-process model is required" -- it needs a live model
        # for the progress-check generate(). The auto-route above cannot rescue it
        # because it only fires for models in NEITHER static list. Running the cell
        # from the model's own conda env is exactly the case where the static
        # subprocess listing is wrong, so promote it there. Guarded on
        # in_process_models being empty: any ensemble that already has an
        # in-process member keeps its existing routing (and its memory profile)
        # byte-for-byte, so this cannot perturb a cell that is mid-flight.
        if not self.in_process_models:
            _here = os.path.basename(sys.prefix)
            _local = [m for m in self.subprocess_models
                      if CONDA_ENVS.get(m) == _here]
            if _local:
                self.in_process_models = _local
                self.subprocess_models = [m for m in self.subprocess_models
                                          if m not in _local]
                self.log(f"  Promoted to in-process (own env '{_here}'): {_local}")

        _covered = set(self.in_process_models) | set(self.subprocess_models)
        _missing = [m for m in self.ensemble_models if m not in _covered]
        if _missing:
            raise ValueError(
                f"ensemble members {_missing} could not be routed to in-process or "
                f"subprocess execution; they would be silently dropped from the "
                f"gradient. Add them to CONDA_ENVS in config.py.")

        # Normalize weights
        active_models = self.in_process_models + self.subprocess_models
        total_weight = sum(self.ensemble_weights.get(m, 1.0) for m in active_models)
        self.norm_weights = {
            m: self.ensemble_weights.get(m, 1.0) / total_weight
            for m in active_models
        }

        self.log(f"Ensemble models: {active_models}")
        self.log(f"  In-process: {self.in_process_models}")
        self.log(f"  Subprocess: {self.subprocess_models}")
        self.log(f"  Weights: {self.norm_weights}")

        # Pre-warm out-of-env workers BEFORE anything else touches the GPU.
        # PyTorch's caching allocator does not give memory back, so once the
        # in-process victims have run a backward pass the parent can be holding
        # ~38 GB and a worker started lazily at the first ensemble step OOMs
        # during weight load. Claiming the worker's memory first makes startup
        # deterministic instead of dependent on allocator state.
        # (Better still: give the worker its own card via ENSEMBLE_WORKER_GPU.)
        if os.environ.get("ENSEMBLE_PERSISTENT_WORKER", "0") == "1":
            for _m in self.subprocess_models:
                if self._get_persistent_worker(_m) is None:
                    raise RuntimeError(
                        f"persistent worker for {_m} failed to start; refusing to "
                        f"run a degraded ensemble silently. See the [stderr] lines "
                        f"above (usually CUDA OOM -> use ENSEMBLE_WORKER_GPU with "
                        f"--gres=gpu:2, or a larger card)."
                    )

        # Initialize EnCodec
        self._init_encodec(encodec_bandwidth)

        # Load in-process models
        self.models: Dict[str, object] = {}
        for model_name in self.in_process_models:
            self.log(f"Loading in-process model: {model_name}...")
            self.models[model_name] = _load_model(model_name, device, dtype)
            # Gradient checkpointing (opt-in via ENSEMBLE_GRAD_CKPT=1): two 7B audio
            # LLMs + a backward pass exceed a 40GB GPU. Checkpointing trades compute
            # for activation memory so the ensemble fits on smaller cards. On an
            # 80GB A100 it is unnecessary (and slower), so it defaults OFF. The attack
            # differentiates w.r.t. the input (not params), so use_reentrant=False +
            # train() (needed for HF to actually checkpoint) is safe here.
            hf = getattr(self.models[model_name], "model", None)

            # FREEZE THE PARAMETERS. This attack differentiates w.r.t. the INPUT
            # waveform; the model weights are constant. But loss.backward() also
            # accumulates a .grad on every parameter, and those buffers are the
            # same size as the weights and are never released -- so each ensemble
            # member silently costs TWICE its weight footprint from its first
            # backward onward. Measured on the N=3 pool (46.5 GB of weights):
            #     after qwen2_audio's first backward   60.6 GB live
            #     after qwen25_omni's first backward   76.0 GB live
            #     gemma3n's first backward             OOM
            # i.e. the pool died at N=3 on an 80 GB card entirely on garbage.
            # Freezing changes no computed value -- the input gradient is
            # identical -- it only stops autograd from materializing gradients
            # nobody reads. Gradient checkpointing with use_reentrant=False still
            # works, because the INPUT carries requires_grad.
            if hf is not None:
                _n = 0
                for _prm in hf.parameters():
                    if _prm.requires_grad:
                        _prm.requires_grad_(False)
                        _n += 1
                    _prm.grad = None
                self.log(f"  Froze {_n} parameter tensors of {model_name} "
                         f"(input-only gradients)")

            if hf is not None and os.environ.get("ENSEMBLE_GRAD_CKPT", "0") == "1":
                try:
                    if hasattr(hf, "config"):
                        hf.config.use_cache = False
                    hf.gradient_checkpointing_enable(
                        gradient_checkpointing_kwargs={"use_reentrant": False})
                    hf.train()
                    # Zero all dropout so train() mode stays deterministic for the attack.
                    for _m in hf.modules():
                        if isinstance(_m, torch.nn.Dropout):
                            _m.p = 0.0
                    # VERIFY, don't assume. hf.gradient_checkpointing_enable() is a
                    # no-op on a model whose modules don't support it, and the
                    # failure is invisible: the run just carries that model's full
                    # activations and OOMs at a pool size the budget said would
                    # fit. Checked here because the alternative is diagnosing it
                    # from a CUDA OOM three models deep.
                    if not any(getattr(_mod, "gradient_checkpointing", False)
                               for _mod in hf.modules()):
                        raise RuntimeError(
                            f"ENSEMBLE_GRAD_CKPT=1 but no module of {model_name} "
                            f"has gradient_checkpointing set: the call was a no-op "
                            f"and this model will carry full activations.")
                    self.log(f"  Gradient checkpointing ENABLED for {model_name}")
                except Exception as e:
                    # FATAL-path output: the ladder builds the attacker with
                    # verbose=False, so self.log() here would swallow the one
                    # explanation of why the pool does not fit.
                    print(f"  ERROR: grad-checkpoint enable failed for "
                          f"{model_name}: {e}", file=sys.stderr, flush=True)
                    raise
            if self.cpu_offload:
                # Park it the moment it is built. Otherwise the CONSTRUCTOR's own
                # peak is all N models resident -- which is precisely the ceiling
                # cpu_offload exists to remove, and which OOMs at N=3 on an 80 GB
                # A100 (measured: gemma3n's forward, 79.2/79.25 GiB in use).
                _hf = getattr(self.models[model_name], "model", None)
                if _hf is None:
                    raise RuntimeError(
                        f"cpu_offload: wrapper for {model_name} exposes no .model "
                        f"module, so it cannot be parked on CPU. Offloading it "
                        f"would be a no-op and the run would OOM at the size "
                        f"cpu_offload was asked to prevent.")
                _hf.to("cpu")
                torch.cuda.empty_cache()
            self.log(f"  Loaded {model_name}")

        self._resident = None if self.cpu_offload else "__all__"

        # Primary model reference (for generate() during attack checks)
        if primary_model in self.models:
            self.target_model = self.models[primary_model]
        elif self.in_process_models:
            self.target_model = self.models[self.in_process_models[0]]
            # RECONCILE THE NAME TOO. primary_model defaults to the global
            # TARGET_MODEL (qwen2_audio); when the pool does not contain it --
            # every leave-one-out cell that holds qwen2_audio OUT -- target_model
            # fell back here but self.primary_model kept the stale name. The
            # warmup filter is
            #     active_in_process = [m for m in in_process if m == primary_model]
            # so it selected NOTHING for all warmup_steps: zero contributing
            # models, total_behavior_loss == 0.0, which is lower than any real CE
            # loss, so best_loss latched at 0.0 on step 1 and best_delta froze at
            # the near-initial random perturbation. The run then returned an
            # essentially UNATTACKED carrier while reporting success-shaped output.
            # This is why every loo_qwen2_audio cell had final_loss == 0.0 on all
            # 50 clips, and why the qwen25_omni ceiling never moved off its clean
            # response.
            self.primary_model = self.in_process_models[0]
            print(f"  NOTE: primary_model {primary_model!r} is not in this "
                  f"ensemble; using {self.primary_model!r} as primary for warmup "
                  f"and progress checks.", file=sys.stderr, flush=True)
        else:
            raise ValueError("At least one in-process model is required")

        # Gradient worker script path
        self._worker_script = os.path.join(SCRIPTS_DIR, "gradient_worker.py")

    def _init_encodec(self, bandwidth: float):
        """Initialize EnCodec model."""
        from encodec_wrapper import EnCodecWrapper
        self.codec = EnCodecWrapper(bandwidth=bandwidth, device=self.device)
        self.log(f"EnCodec loaded (bandwidth={bandwidth} kbps)")

    def log(self, msg: str):
        if self.verbose:
            print(msg)

    def warn(self, msg: str, key: str = "", every: int = 200):
        """A correctness warning that survives verbose=False.

        Every batch driver constructs the attacker with verbose=False, so the
        worker-failure paths below -- which all used self.log() -- were silent.
        That is how 487/950 audio_flamingo gradient requests could fail in a
        ladder cell with the REASON recorded nowhere: the count was printed at
        the end of attack(), but the message saying WHY was suppressed. Print to
        stderr regardless of verbosity, rate-limited so a per-step failure does
        not write a 950-line log per clip.
        """
        if not hasattr(self, "_warn_counts"):
            self._warn_counts = {}
        k = key or msg
        n = self._warn_counts.get(k, 0) + 1
        self._warn_counts[k] = n
        if n <= 3 or n % every == 0:
            print(f"  WARNING[x{n}]: {msg}", file=sys.stderr, flush=True)
        if self.verbose:
            print(f"  WARNING: {msg}")

    def encode_music(self, music_wav: torch.Tensor) -> torch.Tensor:
        """Encode music to EnCodec continuous latent space."""
        music_wav = music_wav.to(self.device)
        with torch.no_grad():
            z = self.codec.encode_to_continuous(music_wav)
        self.log(f"Encoded to latent space: {z.shape}")
        return z

    # ---- persistent subprocess workers -------------------------------------
    # The original _compute_subprocess_gradient below spawns a process and
    # RELOADS the model every step (AF3 load ~46 s => ~600 GPU-h for a full
    # ensemble run). These helpers keep one worker alive per out-of-env model.
    # Opt in with ENSEMBLE_PERSISTENT_WORKER=1; the per-step path is untouched
    # so existing behaviour is byte-identical when the flag is off.

    def _set_resident(self, keep: str):
        """Park every in-process surrogate except `keep` in host RAM.

        Only meaningful under grad_cache: with a cache, exactly one model is
        needed per step, so resident VRAM is one model + EnCodec no matter how
        large the pool gets. A 7B bf16 model is ~14 GB across PCIe (~1-2 s),
        which cache_block amortizes; a reload from disk (~46 s for AF3) would
        not be, which is why this offloads to CPU rather than destroying the
        wrapper. Out-of-env members are unaffected -- they live in their own
        process and should be given their own card via ENSEMBLE_WORKER_GPU.
        """
        if not getattr(self, "cpu_offload", False):
            return
        if getattr(self, "_resident", None) == keep:
            return
        for name, wrapper in self.models.items():
            hf = getattr(wrapper, "model", None)
            if hf is None:
                continue
            want = self.device if name == keep else "cpu"
            try:
                if str(next(hf.parameters()).device).split(":")[0] != str(want).split(":")[0]:
                    hf.to(want)
            except StopIteration:
                continue
        torch.cuda.empty_cache()
        self._resident = keep

    def _get_persistent_worker(self, model_name: str):
        """Spawn (once) and return a live gradient worker for model_name."""
        if not hasattr(self, "_workers"):
            self._workers = {}
        if model_name in self._workers:
            return self._workers[model_name]

        conda_env = CONDA_ENVS.get(model_name)
        if conda_env is None:
            self.warn(f"No conda env for {model_name}", "nocondaenv")
            return None

        conda_exe = os.environ.get("CONDA_EXE", "")
        if conda_exe:
            conda_base = os.path.dirname(os.path.dirname(conda_exe))
        else:
            prefix = os.path.abspath(sys.prefix)
            conda_base = (os.path.dirname(os.path.dirname(prefix))
                          if os.path.basename(os.path.dirname(prefix)) == "envs"
                          else prefix)
        activate = os.path.join(conda_base, "etc", "profile.d", "conda.sh")
        script = os.path.join(SCRIPTS_DIR, "gradient_worker_persistent.py")

        # A dedicated GPU for the worker if one was provided, else share.
        env = {**os.environ}
        worker_gpu = os.environ.get("ENSEMBLE_WORKER_GPU")
        if worker_gpu:
            visible = [x for x in env.get("CUDA_VISIBLE_DEVICES", "").split(",") if x]
            try:
                idx = int(worker_gpu)
                env["CUDA_VISIBLE_DEVICES"] = visible[idx] if len(visible) > idx else worker_gpu
            except ValueError:
                env["CUDA_VISIBLE_DEVICES"] = worker_gpu

        # Call the env's interpreter by ABSOLUTE PATH, never by bare `python`.
        # `conda activate` does NOT reliably win the PATH race: submit_pool_ladder.sh
        # prepends codec-attack/bin to PATH (so DifferentiableOpusProxy can find
        # ffmpeg), and conda only removes PATH entries it added itself. Measured:
        #     conda activate audio-llm -> CONDA_PREFIX=.../envs/audio-llm   (ok)
        #                                 which python =.../codec-attack/bin/python (!)
        # So every out-of-env worker silently ran under codec-attack: phi4 got
        # peft 0.18.1 instead of 0.11.1 ("Phi4MMModel has no attribute
        # prepare_inputs_for_generation") and audio_flamingo got transformers
        # 4.57 instead of 5.0.0.dev0. Both failed, were skipped, and contributed
        # ZERO gradient to every rung of the pool ladder while the run reported
        # a healthy ensemble.
        # conda activate is still sourced first, for env vars the model needs
        # (LD_LIBRARY_PATH etc.); the absolute path just makes the interpreter
        # immune to whatever PATH the caller set.
        env_python = os.path.join(conda_base, "envs", conda_env, "bin", "python")
        cmd = persistent_worker_command(model_name, conda_base, script)
        # Worker stderr goes to a FILE, never DEVNULL: this is the only place the
        # reason for a failed start is recorded, and a worker that fails to start
        # silently degrades the run to a smaller ensemble.
        # Durable, SHARED stderr path. tempfile.gettempdir() on a SLURM node is
        # node-local and is reaped when the job ends, so the one recording of why
        # a worker died vanished exactly when it was needed. Keep it beside the
        # job logs, tagged with the job id so concurrent cells cannot clobber
        # each other's file.
        _errdir = os.path.join(PROJECT_ROOT, "logs", "workers")
        try:
            os.makedirs(_errdir, exist_ok=True)
        except Exception:
            _errdir = tempfile.gettempdir()
        _jid = os.environ.get("SLURM_JOB_ID", str(os.getpid()))
        err_path = os.path.join(_errdir, f"worker_{model_name}_{_jid}.err")
        err_fh = open(err_path, "w")
        self._worker_err_paths = getattr(self, "_worker_err_paths", {})
        self._worker_err_paths[model_name] = err_path
        self.log(f"  Starting persistent worker for {model_name} "
                 f"(env={conda_env}, gpu={env.get('CUDA_VISIBLE_DEVICES','inherit')}, "
                 f"stderr={err_path})")
        proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=err_fh, text=True, bufsize=1, env=env,
        )
        # Block until the model is resident. A worker that dies during load must
        # fail loudly here rather than silently contributing zero gradients.
        # Skip any banner lines a model wrapper prints to stdout before READY
        # (the worker redirects those to stderr, but don't rely on it).
        handshake = None
        for _ in range(200):
            line = proc.stdout.readline()
            if not line:            # EOF => worker died during load
                break
            line = line.strip()
            if line == "READY":
                handshake = line
                break
            self.log(f"  [worker {model_name}] {line[:120]}")
        if handshake != "READY":
            # This diagnostic is FATAL-path output and must NOT be gated on
            # self.log: every ladder cell constructs the attacker with
            # verbose=False, which silenced the only explanation of the failure
            # and left "worker failed to start" with no cause attached.
            print(f"  ERROR: worker for {model_name} failed to start "
                  f"(stderr file: {err_path})", file=sys.stderr, flush=True)
            try:
                err_fh.flush()
                with open(err_path) as fh:
                    tail = fh.read().strip().splitlines()[-25:]
                for ln in tail:
                    print(f"    [stderr] {ln[:300]}", file=sys.stderr, flush=True)
            except Exception as _e:
                print(f"    [stderr] unreadable: {_e}", file=sys.stderr, flush=True)
            proc.kill()
            return None
        self.log(f"  Worker for {model_name} READY")

        tmpdir = tempfile.mkdtemp(prefix=f"ensw_{model_name}_")
        self._workers[model_name] = (proc, tmpdir)
        return self._workers[model_name]

    def _compute_persistent_gradient(
        self,
        model_name: str,
        audio_16k: torch.Tensor,
        target_text: str,
        prompt: str = None,
    ) -> Tuple[Optional[torch.Tensor], float]:
        """Gradient from a long-lived worker. Returns (None, 0.0) on failure."""
        w = self._get_persistent_worker(model_name)
        if w is None:
            return None, 0.0
        proc, tmpdir = w

        req = os.path.join(tmpdir, "req.pt")
        # The scratch dir lives in the node-local (job-scoped) /tmp and has been
        # observed to disappear mid-run -- job 62193075 died at clip 72/224 with
        # "Parent directory /tmp/ensw_audio_flamingo_* does not exist". Recreate
        # it, and treat a still-failing save as a COUNTED skip (see _subproc_skips
        # in attack()) rather than letting it kill the whole benchmark.
        try:
            os.makedirs(tmpdir, exist_ok=True)
            torch.save({"audio_16k": audio_16k.detach().cpu(),
                        "target_text": target_text,
                        "prompt": prompt or ""}, req)
        except OSError as e:
            self.warn(f"worker {model_name} request write failed: {e}", "reqwrite")
            self._workers.pop(model_name, None)
            return None, 0.0
        try:
            proc.stdin.write(req + "\n")
            proc.stdin.flush()
            resp = proc.stdout.readline().strip()
        except (BrokenPipeError, ValueError) as e:
            self.warn(f"worker {model_name} pipe broke: {e}", "pipe")
            self._workers.pop(model_name, None)
            return None, 0.0

        if not resp.startswith("OK"):
            self.warn(f"worker {model_name} returned {resp!r}", "resp")
            return None, 0.0

        out = req + ".out"
        if not os.path.isfile(out):
            self.warn(f"worker {model_name} wrote no output file", "noout")
            return None, 0.0
        try:
            result = torch.load(out, map_location="cpu", weights_only=True)
        except (OSError, RuntimeError) as e:
            # Same scratch-dir failure mode, on the read side.
            self.warn(f"worker {model_name} response read failed: {e}", "respread")
            self._workers.pop(model_name, None)
            return None, 0.0
        return result["gradient"].to(self.device), float(result["loss"])

    def close_workers(self):
        """Shut down any persistent workers (safe to call repeatedly)."""
        for name, (proc, tmpdir) in getattr(self, "_workers", {}).items():
            try:
                proc.stdin.write("QUIT\n")
                proc.stdin.flush()
                proc.wait(timeout=30)
            except Exception:
                proc.kill()
            self.log(f"  Worker for {name} stopped")
        self._workers = {}

    def _compute_subprocess_gradient(
        self,
        model_name: str,
        audio_16k: torch.Tensor,
        target_text: str,
        prompt: str = None,
    ) -> Tuple[Optional[torch.Tensor], float]:
        """
        Compute gradient for a model running in a separate conda env.

        Saves audio tensor to a temp file, invokes gradient_worker.py in the
        appropriate conda env, and reads back the gradient tensor.

        Returns:
            (gradient, loss_value) or (None, 0.0) on failure
        """
        conda_env = CONDA_ENVS.get(model_name)
        if conda_env is None:
            self.log(f"  WARNING: No conda env for {model_name}, skipping")
            return None, 0.0

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "audio_16k.pt")
            output_path = os.path.join(tmpdir, "gradient.pt")

            # Save input audio (detached, on CPU)
            torch.save({
                "audio_16k": audio_16k.detach().cpu(),
                "target_text": target_text,
                "prompt": prompt or "",
                "model_name": model_name,
            }, input_path)

            # Find conda base
            conda_base = os.environ.get("CONDA_EXE", "")
            if conda_base:
                conda_base = os.path.dirname(os.path.dirname(conda_base))
            else:
                prefix = os.path.abspath(sys.prefix)
                conda_base = (os.path.dirname(os.path.dirname(prefix))
                              if os.path.basename(os.path.dirname(prefix)) == "envs"
                              else prefix)

            activate_script = os.path.join(conda_base, "etc", "profile.d", "conda.sh")

            # ABSOLUTE interpreter path -- see the long note in
            # _get_persistent_worker. A bare `python` here resolves against a PATH
            # the caller may have prepended, so audio_flamingo and phi4 ran under
            # codec-attack and failed on EVERY call: 449-508 skips per clip in the
            # N=2 cells, 604-647 in N=3, matching the expected number of EoT picks
            # exactly. Neither model ever contributed a gradient.
            env_python = os.path.join(conda_base, "envs", conda_env, "bin", "python")
            cmd = (
                f'source "{activate_script}" && '
                f'conda activate {conda_env} && '
                f'"{env_python}" "{self._worker_script}" '
                f'--input "{input_path}" '
                f'--output "{output_path}"'
            )

            try:
                proc = subprocess.run(
                    ["bash", "-c", cmd],
                    capture_output=True,
                    text=True,
                    timeout=120,
                    env={**os.environ, "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0")},
                )

                if proc.returncode != 0:
                    self.log(f"  WARNING: Subprocess {model_name} failed (rc={proc.returncode})")
                    if proc.stderr:
                        self.log(f"  stderr: {proc.stderr[:200]}")
                    # Announce the FIRST failure per model on stderr regardless of
                    # verbose. Ladder cells construct with verbose=False, so this
                    # warning went nowhere while ~23,760 failed calls per cell were
                    # skipped-and-rescaled into a silently degraded ensemble. Only
                    # the first is printed -- it can fire ~500x per clip.
                    self._announced = getattr(self, "_announced", set())
                    if model_name not in self._announced:
                        self._announced.add(model_name)
                        print(f"  ERROR: subprocess gradient for {model_name} FAILED "
                              f"(rc={proc.returncode}); it will contribute NO gradient. "
                              f"Further failures suppressed; see _subproc_skips.",
                              file=sys.stderr, flush=True)
                        for ln in (proc.stderr or "").strip().splitlines()[-15:]:
                            print(f"    [stderr] {ln[:300]}", file=sys.stderr, flush=True)
                    return None, 0.0

                if not os.path.isfile(output_path):
                    self.log(f"  WARNING: No output file from {model_name}")
                    return None, 0.0

                result = torch.load(output_path, map_location="cpu", weights_only=True)
                grad = result["gradient"].to(self.device)
                loss_val = result["loss"]
                return grad, loss_val

            except subprocess.TimeoutExpired:
                self.log(f"  WARNING: Subprocess {model_name} timed out")
                return None, 0.0
            except Exception as e:
                self.log(f"  WARNING: Subprocess {model_name} error: {e}")
                return None, 0.0

    def _compute_in_process_gradient(
        self,
        model_name: str,
        audio_16k: torch.Tensor,
        target_text: str,
        prompt: str = None,
    ) -> Tuple[torch.Tensor, float]:
        """
        Compute gradient for an in-process model.

        audio_16k must have requires_grad=True (or be a leaf that we can
        get .grad from). We compute loss and call backward to populate
        audio_16k.grad.

        Returns:
            (gradient w.r.t. audio_16k, loss_value)
        """
        model = self.models[model_name]

        # Reset model caches if needed
        if hasattr(model, 'reset_cache'):
            model.reset_cache()

        # Force train() at the call site so HF gradient checkpointing actually
        # activates (the wrapper loads the model in eval(); HF only checkpoints
        # when training=True). Only when checkpointing is opted in; dropout was
        # zeroed at load, so this stays deterministic.
        if os.environ.get("ENSEMBLE_GRAD_CKPT", "0") == "1":
            hf = getattr(model, "model", None)
            if hf is not None:
                hf.train()

        audio_input = audio_16k.detach().requires_grad_(True)
        loss = model.compute_loss(audio_input, target_text, prompt=prompt)
        loss.backward()

        if audio_input.grad is None:
            raise RuntimeError(
                f"No gradient for {model_name}: loss.requires_grad={loss.requires_grad}, "
                f"loss.grad_fn={loss.grad_fn}, audio_input.is_leaf={audio_input.is_leaf}"
            )

        grad = audio_input.grad.detach().clone()
        loss_val = loss.item()

        return grad, loss_val

    def attack(
        self,
        music_wav: torch.Tensor,
        target_text: str,
        steps: int = ATTACK_STEPS,
        check_every: int = CHECK_EVERY,
        music_name: str = "",
        use_multi_scale_loss: bool = False,
        prompt: str = None,
        untargeted: bool = False,
    ) -> AttackResult:
        """
        Run ensemble latent-space adversarial attack.

        Uses gradient stitching: forward through EnCodec decoder once,
        then accumulate per-model gradients independently and chain back
        through the decoder via audio_16k.backward(combined_grad).

        Args:
            music_wav: Music carrier [1, 1, T] at 24kHz
            target_text: Text to force Audio LLMs to output
            steps: Number of optimization steps
            check_every: Check model output every N steps
            music_name: Name of the music file (for logging)
            use_multi_scale_loss: Use multi-scale mel loss
            prompt: Text prompt for the models
            untargeted: If True, maximize loss on original output instead
                of minimizing loss on target_text.

        Returns:
            AttackResult with adversarial audio and metrics
        """
        start_time = time.time()

        music_wav = music_wav.to(self.device)

        # Reset all in-process model caches
        for m in self.models.values():
            if hasattr(m, 'reset_cache'):
                m.reset_cache()

        # Per-attack counter. It used to accumulate across clips while the
        # end-of-attack warning compared it against a SINGLE attack's expected
        # request count, so a batch driver over-reported failures on later clips
        # and no caller could attribute a failure to the clip it happened on.
        self._subproc_skips = 0
        self._empty_steps = 0

        # Encode to latent space
        z_original = self.encode_music(music_wav)

        # Cache original decoded audio for perceptual loss
        with torch.no_grad():
            original_audio_24k = self.codec.decode_from_continuous(z_original)
            original_audio_16k = torchaudio.functional.resample(
                original_audio_24k.squeeze(0), ENCODEC_SAMPLE_RATE, TARGET_SAMPLE_RATE
            )
            self._set_resident(self.primary_model)
            original_output = self.target_model.generate(original_audio_16k, prompt=prompt)

        self.log(f"Original model output: {original_output}")

        # Untargeted mode: disrupt the original output instead of forcing a target
        if untargeted:
            target_text = original_output
            self.log(f"UNTARGETED mode: maximizing loss on original output")
        self.log(f"Target: {target_text}")
        if prompt:
            self.log(f"Prompt: {prompt}")

        # Initialize learnable perturbation.
        # Default stays zeros-init so every previously produced bundle remains
        # reproducible byte-for-byte. Random init is opt-in and exists ONLY to
        # make a seed sweep meaningful: with zeros-init this attack is fully
        # deterministic, so re-running under different SEED values yields
        # identical bundles and a "3-seed" table would be one run reported
        # three times. Mirrors the rand-init in robust_latent_attack.py.
        if self.rand_init:
            gen = torch.Generator(device="cpu").manual_seed(int(self.seed))
            noise = torch.randn(z_original.shape, generator=gen,
                                dtype=torch.float32).to(z_original.device)
            # Scale relative to the latent's own magnitude so the init is a small
            # perturbation regardless of eps units, then clamp into the eps ball.
            noise = noise / noise.norm() * (self.rand_init_scale * z_original.norm())
            noise = noise.clamp(-self.eps, self.eps)
            delta = noise.to(z_original.dtype).clone().requires_grad_(True)
            self.log(f"Random init: seed={self.seed} "
                     f"scale={self.rand_init_scale} |delta_0|={delta.norm().item():.4f}")
        else:
            delta = torch.zeros_like(z_original, requires_grad=True)
        optimizer = torch.optim.Adam([delta], lr=self.alpha)

        # History tracking
        history = {
            "loss": [], "behavior_loss": [], "perceptual_loss": [],
            "outputs": [], "steps": [],
            "per_model_loss": {m: [] for m in self.ensemble_models},
            "gradient_alignment": [],
            "grad_staleness": [],
        }
        # Opt-in observations only: preserve the default update and checkpoint behavior.
        self.diagnostic_checkpoints = {}
        diagnostic_best = {'all_steps': float('inf'), 'eligible': float('inf')}
        diagnostic_eligible = max(self.warmup_steps, int(self.channel_warmup_ratio * steps) if self.opus_proxy is not None or self.mp3_proxy is not None else 0)
        history['diagnostic'] = []
        history['raw_gradient_norms'] = []
        best_delta = delta.data.clone()
        best_loss = float('inf')
        best_step = None
        success = False
        best_eligible_from = 0
        if self.best_after_warmup:
            best_eligible_from = self.warmup_steps
            if self.opus_proxy is not None or self.mp3_proxy is not None:
                best_eligible_from = max(best_eligible_from,
                                         int(self.channel_warmup_ratio * steps))
            self.log(f"best_delta selection restricted to step >= {best_eligible_from}")

        # PER-CLIP counters. These were instance attributes accumulated with
        # getattr(..., 0) + 1, so in a batch driver (50 clips through one
        # attacker) the end-of-attack warning reported the running total for the
        # whole job, not this clip -- and a clip that was fine still printed a
        # scary number inherited from an earlier one. Reset here so every count
        # belongs to exactly one clip.
        self._subproc_skips = 0
        self._empty_steps = 0
        self._contrib_steps = 0

        all_models = self.in_process_models + self.subprocess_models

        # Per-CLIP gradient cache. Reset here, never across clips: the cached
        # tensors are shaped by this carrier's length and a stale entry carried
        # into the next clip would be both wrong and silent.
        self._gcache = {}        # model -> unit-normalized grad [1, T] on device
        self._gcache_loss = {}   # model -> float
        self._gcache_step = {}   # model -> step index it was refreshed at
        # A member that misses more than cache_ttl_mult full cycles is a dead
        # worker, not a stale gradient. The pool-ladder postmortem is exactly
        # this failure with a different mechanism (a dead worker emptied the
        # step and best_loss latched at 0.0 for the rest of the run), so this
        # raises rather than dropping the member and rescaling in silence.
        self._cache_ttl = int(self.cache_ttl_mult * self.cache_block * max(1, len(all_models))) + 1
        if self.grad_cache:
            self.log(f"Gradient cache ON: block={self.cache_block} "
                     f"pool={len(all_models)} ttl={self._cache_ttl} steps "
                     f"cpu_offload={self.cpu_offload}")

        self.log(f"\nStarting ensemble attack: eps={self.eps}, alpha={self.alpha}, steps={steps}")
        self.log(f"Mode: {'untargeted' if untargeted else 'targeted'}")
        self.log(f"Perceptual weight: {self.perceptual_weight}")
        self.log(f"Warmup steps (primary only): {self.warmup_steps}")
        self.log(f"Gradient normalize: {self.grad_normalize}")
        self.log("-" * 60)

        # Enable gradients through EnCodec decoder
        self.codec.model.decoder.train()

        for step in range(steps):
            optimizer.zero_grad()

            # Apply perturbation in latent space
            z_adv = z_original + delta

            # Decode to audio (24kHz) — shared forward pass
            audio_adv_24k = self.codec.decode_from_continuous(z_adv)

            # Multi-bitrate Opus EoT (STE/BPDA). Two-stage schedule mirroring
            # robust_latent_attack: a clean warm-up so the attack first finds the
            # target, then codec-hardened steps for the rest of the budget.
            if self.opus_proxy is not None:
                if step >= int(self.channel_warmup_ratio * steps):
                    audio_adv_24k = self.opus_proxy(audio_adv_24k)
            elif self.mp3_proxy is not None:
                if step >= int(self.channel_warmup_ratio * steps):
                    audio_adv_24k = self.mp3_proxy(audio_adv_24k)

            # Resample to 16kHz — this is the stitching point
            audio_adv_16k = torchaudio.functional.resample(
                audio_adv_24k.squeeze(0), ENCODEC_SAMPLE_RATE, TARGET_SAMPLE_RATE
            )

            # Decide which models are active this step
            if step < self.warmup_steps:
                active_in_process = [m for m in self.in_process_models if m == self.primary_model]
                active_subprocess = []
            elif self.grad_cache:
                # Refresh ONE surrogate this step; every other member still
                # contributes through its cached gradient below. The first
                # len(pool) post-warmup steps walk the pool in order so the
                # cache is filled before any member's entry can go stale.
                pool = list(self.in_process_models) + list(self.subprocess_models)
                t = step - self.warmup_steps
                if t < len(pool):
                    pick = pool[t]
                else:
                    pick = pool[((t - len(pool)) // self.cache_block) % len(pool)]
                active_in_process = [pick] if pick in self.in_process_models else []
                active_subprocess = [pick] if pick in self.subprocess_models else []
                if self.cpu_offload:
                    self._set_resident(pick)
            elif self.model_eot:
                # EoT over the victim set: sample ONE model per step instead of
                # averaging all of them, mirroring how the codec dimension is
                # handled. Same expected objective as the mean, but ~1/N the
                # per-step cost, and the sampling noise stops the optimizer from
                # spending budget maintaining an already-solved victim (gradients
                # are unit-normalized, so a converged model otherwise keeps
                # pulling at full weight).
                pool = list(self.in_process_models) + list(self.subprocess_models)
                pick = pool[self._eot_rng.randrange(len(pool))]
                active_in_process = [pick] if pick in self.in_process_models else []
                active_subprocess = [pick] if pick in self.subprocess_models else []
            else:
                active_in_process = list(self.in_process_models)
                active_subprocess = list(self.subprocess_models)

            # Compute perceptual loss first with retain_graph=True
            # (model-independent, computed at audio_24k level)
            if self.perceptual_weight > 0:
                if use_multi_scale_loss:
                    p_loss = multi_scale_mel_distance(
                        audio_adv_24k.squeeze(0), original_audio_24k.squeeze(0).detach(),
                        sample_rate=ENCODEC_SAMPLE_RATE
                    )
                else:
                    p_loss = mel_distance(
                        audio_adv_24k.squeeze(0), original_audio_24k.squeeze(0).detach(),
                        sample_rate=ENCODEC_SAMPLE_RATE
                    )
                # Backprop perceptual loss (retain graph for behavior grads)
                (self.perceptual_weight * p_loss).backward(retain_graph=True)
                p_loss_val = p_loss.item()
            else:
                p_loss_val = 0.0

            # --- Gradient stitching ---
            # Collect per-model gradients w.r.t. audio_16k
            combined_grad = torch.zeros_like(audio_adv_16k.detach())
            total_behavior_loss = 0.0
            active_weight_sum = 0.0
            step_grads = {}
            step_losses = {}
            raw_norms = {}

            for model_name in active_in_process:
                grad, loss_val = self._compute_in_process_gradient(
                    model_name, audio_adv_16k, target_text, prompt=prompt
                )
                w = self.norm_weights.get(model_name, 1.0)

                if self.diagnostic_interval:
                    raw_norms[model_name] = float(grad.norm().item())
                if self.grad_normalize:
                    grad_norm = grad.norm()
                    if grad_norm > 1e-8:
                        grad = grad / grad_norm

                step_grads[model_name] = grad.detach()
                step_losses[model_name] = loss_val

                combined_grad += w * grad
                total_behavior_loss += w * loss_val
                active_weight_sum += w
                history["per_model_loss"][model_name].append(loss_val)

            for model_name in active_subprocess:
                if os.environ.get("ENSEMBLE_PERSISTENT_WORKER", "0") == "1":
                    grad, loss_val = self._compute_persistent_gradient(
                        model_name, audio_adv_16k, target_text, prompt=prompt
                    )
                else:
                    grad, loss_val = self._compute_subprocess_gradient(
                        model_name, audio_adv_16k, target_text, prompt=prompt
                    )
                if grad is None:
                    # Skip failed subprocess, rescale remaining weights.
                    # NOTE: this is silent-by-design in the original code. Count
                    # the skips so a run where the out-of-env victim contributed
                    # nothing cannot be mistaken for a successful ensemble.
                    self._subproc_skips = getattr(self, "_subproc_skips", 0) + 1
                    if self.strict_workers:
                        raise RuntimeError(
                            f"strict ensemble: surrogate {model_name} failed to "
                            "return a gradient; refusing a degraded transfer bundle")
                    continue

                w = self.norm_weights.get(model_name, 1.0)

                if self.diagnostic_interval:
                    raw_norms[model_name] = float(grad.norm().item())
                if self.grad_normalize:
                    grad_norm = grad.norm()
                    if grad_norm > 1e-8:
                        grad = grad / grad_norm

                step_grads[model_name] = grad.detach()
                step_losses[model_name] = loss_val

                combined_grad += w * grad
                total_behavior_loss += w * loss_val
                active_weight_sum += w
                history["per_model_loss"][model_name].append(loss_val)

            # --- Fold this step's fresh gradients into the cache ---------
            if self.grad_cache:
                for _name, _g in step_grads.items():
                    _prev = self._gcache.get(_name)
                    if _prev is not None and _prev.shape == _g.shape:
                        # Free measurement of the assumption the cache rests on:
                        # how far did THIS model's gradient rotate over the k
                        # steps since it was last refreshed? If this decays to
                        # the cross-model floor within k < N, the cache is not
                        # valid -- and neither is any claim that pool size
                        # matters, because the members are not pulling in a
                        # stable direction to begin with.
                        _a, _b = _prev.reshape(-1).float(), _g.reshape(-1).float()
                        _den = _a.norm() * _b.norm()
                        history["grad_staleness"].append({
                            "step": step, "model": _name,
                            "k": step - self._gcache_step[_name],
                            "cosine": float(torch.dot(_a, _b).div(_den).item())
                            if _den > 1e-12 else 0.0,
                        })
                    self._gcache[_name] = _g
                    self._gcache_loss[_name] = step_losses[_name]
                    self._gcache_step[_name] = step

                # Recombine over the WHOLE cache, not just this step's refresh.
                combined_grad = torch.zeros_like(audio_adv_16k.detach())
                total_behavior_loss = 0.0
                active_weight_sum = 0.0
                for _name in all_models:
                    _g = self._gcache.get(_name)
                    if _g is None:
                        continue                      # cache still warming
                    _age = step - self._gcache_step[_name]
                    if _age > self._cache_ttl:
                        raise RuntimeError(
                            f"gradient cache: {_name} last refreshed at step "
                            f"{self._gcache_step[_name]} ({_age} steps ago, ttl="
                            f"{self._cache_ttl}). Its worker is dead, not slow; "
                            f"refusing to keep combining a frozen gradient at "
                            f"full weight. See logs/workers/.")
                    _w = self.norm_weights.get(_name, 1.0)
                    combined_grad += _w * _g
                    total_behavior_loss += _w * self._gcache_loss[_name]
                    active_weight_sum += _w

            # Rescale if some models were skipped
            if active_weight_sum > 0 and active_weight_sum < 1.0:
                combined_grad = combined_grad / active_weight_sum
                total_behavior_loss = total_behavior_loss / active_weight_sum

            # In cache mode only one model is recomputed per step, so cross-model
            # cosines must be read off the cache or they would vanish entirely.
            _align_src = self._gcache if self.grad_cache else step_grads
            names = sorted(_align_src)
            align = {"step": step, "models": names, "cosine": {}, "norm": {}}
            for name in names:
                align["norm"][name] = float(_align_src[name].norm().item())
            for i, ni in enumerate(names):
                for nj in names[i + 1:]:
                    gi, gj = _align_src[ni].reshape(-1), _align_src[nj].reshape(-1)
                    den = gi.norm() * gj.norm()
                    align["cosine"][f"{ni}__{nj}"] = float(torch.dot(gi, gj).div(den).item()) if den > 1e-12 else 0.0
            history["gradient_alignment"].append(align)

            if active_weight_sum <= 0 and self.strict_workers:
                raise RuntimeError(
                    "strict ensemble: no surrogate contributed a gradient on "
                    f"step {step + 1}; refusing an invalid transfer bundle")

            # Untargeted: negate gradients to maximize loss (disrupt original output)
            if untargeted:
                combined_grad = -combined_grad
                total_behavior_loss = -total_behavior_loss

            # Env-gated memory trace. The N=3 pool OOMs ~15 GB above what the
            # per-model budget predicts, and a CUDA OOM traceback says only where
            # the last 32 MB was requested, not what is holding the other 78 GB.
            if os.environ.get("ENSEMBLE_MEM_LOG", "0") == "1":
                print(f"[mem] step={step} active={sorted(step_grads)} "
                      f"alloc={torch.cuda.memory_allocated()/2**30:.1f}G "
                      f"peak={torch.cuda.max_memory_allocated()/2**30:.1f}G "
                      f"reserved={torch.cuda.memory_reserved()/2**30:.1f}G",
                      file=sys.stderr, flush=True)

            if self.diagnostic_interval:
                history['raw_gradient_norms'].append({'step': step + 1, 'norms': raw_norms})
                scored_loss = total_behavior_loss + self.perceptual_weight * p_loss_val
                for rule in ('all_steps', 'eligible'):
                    eligible = rule == 'all_steps' or step >= diagnostic_eligible
                    if eligible and active_weight_sum > 0 and scored_loss < diagnostic_best[rule]:
                        diagnostic_best[rule] = scored_loss
                        # This is the perturbation actually scored, BEFORE optimizer.step().
                        self.diagnostic_checkpoints[rule] = dict(delta=delta.detach().cpu().clone(), step=step+1, loss=scored_loss)
                if step % self.diagnostic_interval == 0 or step == steps - 1:
                    latent_grads = {name: torch.autograd.grad(audio_adv_16k, delta, grad_outputs=g, retain_graph=True)[0].detach().float() for name,g in step_grads.items()}
                    cosines = {}
                    ns = sorted(latent_grads)
                    for i,ni in enumerate(ns):
                        for nj in ns[i+1:]:
                            gi,gj = latent_grads[ni].flatten(),latent_grads[nj].flatten()
                            cosines[ni+'__'+nj] = float(torch.nn.functional.cosine_similarity(gi,gj,dim=0).item())
                    history['diagnostic'].append(dict(step=step+1, latent_cosines=cosines, latent_norms={n:float(g.norm()) for n,g in latent_grads.items()}, delta_linf=float(delta.detach().abs().max()), boundary_fraction=float((delta.detach().abs() >= self.eps * .999).float().mean()), member_losses=dict(step_losses)))
                    del latent_grads

            # Backprop combined gradient through the EnCodec decoder
            # audio_adv_16k is connected to z_adv via resample <- decode <- z_adv
            audio_adv_16k.backward(combined_grad)

            optimizer.step()

            # Project to epsilon ball
            with torch.no_grad():
                delta.data = torch.clamp(delta.data, -self.eps, self.eps)

            # Track
            total_loss = total_behavior_loss + self.perceptual_weight * p_loss_val
            history["loss"].append(total_loss)
            history["behavior_loss"].append(total_behavior_loss)
            history["perceptual_loss"].append(p_loss_val)

            # Only a step where SOMETHING actually contributed may claim to be the
            # best. When no model contributes -- warmup with a primary outside the
            # pool, or every EoT-sampled model failing -- total_behavior_loss is
            # 0.0 by construction, not because the attack succeeded. 0.0 beats every
            # real CE loss, so best_loss latched there and best_delta froze at a
            # near-initial perturbation for the rest of the run, and the bundle was
            # written from an essentially unattacked carrier. Guard on the same
            # active_weight_sum the rescale above uses.
            if active_weight_sum > 0:
                self._contrib_steps += 1
                if total_loss < best_loss and step >= best_eligible_from:
                    best_loss = total_loss
                    best_delta = delta.data.clone()
                    best_step = step + 1
            else:
                self._empty_steps += 1

            # Progress logging
            if (step + 1) % 10 == 0:
                model_losses = " | ".join(
                    f"{m}: {history['per_model_loss'][m][-1]:.4f}"
                    for m in all_models
                    if history["per_model_loss"][m]
                )
                phase = "warmup" if step < self.warmup_steps else "ensemble"
                self.log(
                    f"Step {step+1:4d}/{steps} [{phase}] | "
                    f"Total: {total_loss:.4f} | "
                    f"Behavior: {total_behavior_loss:.4f} | "
                    f"Perceptual: {p_loss_val:.4f} | "
                    f"{model_losses}"
                )

            # Check model output periodically (use primary model)
            if check_every > 0 and (step + 1) % check_every == 0:
                with torch.no_grad():
                    z_test = z_original + delta
                    audio_test_24k = self.codec.decode_from_continuous(z_test)
                    audio_test_16k = torchaudio.functional.resample(
                        audio_test_24k.squeeze(0), ENCODEC_SAMPLE_RATE, TARGET_SAMPLE_RATE
                    )
                    self._set_resident(self.primary_model)
                    output = self.target_model.generate(audio_test_16k, prompt=prompt)

                history["outputs"].append(output)
                history["steps"].append(step + 1)
                self.log(f"  -> Primary output: {output[:100]}")

                if untargeted:
                    wer = compute_wer(target_text, output)
                    if wer > UNTARGETED_WER_THRESHOLD:
                        self.log(f"  -> DISRUPTED at step {step+1}! (WER={wer:.3f} > {UNTARGETED_WER_THRESHOLD})")
                        success = True
                    else:
                        self.log(f"  -> Not disrupted (WER={wer:.3f})")
                else:
                    if target_text.lower() in output.lower():
                        self.log(f"  -> TARGET ACHIEVED at step {step+1}!")
                        success = True

        if self.diagnostic_interval:
            self.diagnostic_checkpoints['final'] = dict(delta=delta.detach().cpu().clone(), step=steps, loss=None)

        # Restore eval mode
        self.codec.model.decoder.eval()

        # Final evaluation with best delta (definitive success check)
        with torch.no_grad():
            z_final = z_original + best_delta
            audio_final_24k = self.codec.decode_from_continuous(z_final)
            audio_final_16k = torchaudio.functional.resample(
                audio_final_24k.squeeze(0), ENCODEC_SAMPLE_RATE, TARGET_SAMPLE_RATE
            )
            self._set_resident(self.primary_model)
            final_output = self.target_model.generate(audio_final_16k, prompt=prompt)

        # Reset success — final output is the definitive answer
        success = False
        if untargeted:
            final_wer = compute_wer(target_text, final_output)
            if final_wer > UNTARGETED_WER_THRESHOLD:
                success = True
            self.log(f"Final WER vs original: {final_wer:.3f} (threshold={UNTARGETED_WER_THRESHOLD})")
        else:
            if target_text.lower() in final_output.lower():
                success = True

        # Compute SNR
        snr_db = LatentCodecAttacker._compute_snr(
            original_audio_24k.squeeze().detach(),
            audio_final_24k.squeeze().detach()
        )
        latent_snr_db = LatentCodecAttacker._compute_snr(
            z_original.detach(), z_final.detach()
        )

        duration = time.time() - start_time

        self.log("\n" + "=" * 60)
        self.log("Ensemble Attack Complete")
        self.log(f"Models: {all_models}")
        self.log(f"Final Loss: {best_loss:.4f}")
        self.log(f"Best checkpoint step: {best_step} (eligible from step {best_eligible_from + 1})")
        self.log(f"Audio SNR: {snr_db:.2f} dB")
        self.log(f"Latent SNR: {latent_snr_db:.2f} dB")
        self.log(f"Success (primary): {success}")
        self.log(f"Final Output: {final_output}")
        self.log(f"Duration: {duration:.1f}s")
        # Loud on the failure mode that would otherwise look like a clean run:
        # an out-of-env victim whose gradients never made it into the ensemble.
        # The failure that produced a whole invalid ladder: zero models
        # contributing, best_loss never leaving inf, best_delta frozen at init.
        # Refuse to hand back a bundle rather than let a driver serialize a
        # success-shaped row from an unattacked carrier.
        if self._contrib_steps == 0:
            raise RuntimeError(
                f"ensemble attack contributed NOTHING on all {steps} steps "
                f"(primary={self.primary_model!r}, in_process={self.in_process_models}, "
                f"subprocess={self.subprocess_models}): best_delta is still the "
                f"initialization, so the returned carrier is unattacked. Refusing "
                f"to return a bundle.")
        if self._empty_steps:
            emsg = (f"!!! WARNING: {self._empty_steps}/{steps} steps had NO "
                    f"contributing model (primary={self.primary_model!r}); only "
                    f"{self._contrib_steps} steps could update best_delta. A run "
                    f"with empty warmup steps optimized for fewer steps than it "
                    f"reports.")
            print(emsg, file=sys.stderr, flush=True)
            self.log(emsg)

        skips = self._subproc_skips
        if skips:
            expected = max(0, steps - self.warmup_steps) * len(self.subprocess_models)
            msg = (f"!!! WARNING: {skips}/{expected} subprocess gradient requests "
                   f"FAILED for {self.subprocess_models} — those victims did NOT "
                   f"fully participate in this attack. Do not report it as an "
                   f"{len(self.ensemble_models)}-model ensemble without checking.")
            # NOT self.log(): this is a correctness alarm, and every batch driver
            # runs with verbose=False, which would silence precisely the message
            # saying the ensemble was smaller than it claims.
            print(msg, file=sys.stderr, flush=True)
            self.log(msg)
        self.log("=" * 60)

        history["best_step"] = best_step
        history["best_eligible_from_step"] = best_eligible_from + 1

        return AttackResult(
            original_wav=original_audio_24k.detach().cpu().squeeze(0),
            adversarial_wav=audio_final_24k.detach().cpu().squeeze(0),
            original_latents=z_original.detach().cpu(),
            adversarial_latents=z_final.detach().cpu(),
            latent_perturbation=best_delta.detach().cpu(),
            original_output=original_output,
            adversarial_output=final_output,
            target_text=target_text,
            success=success,
            final_loss=best_loss,
            steps_taken=steps,
            snr_db=snr_db,
            latent_snr_db=latent_snr_db,
            history=history,
            music_name=music_name,
            attack_duration_s=duration,
        )

    # ------------------------------------------------------------------
    # Delegate robustness testing and saving to LatentCodecAttacker utils
    # ------------------------------------------------------------------

    def test_opus_robustness(
        self,
        result: AttackResult,
        bitrates: List[int] = None,
        prompt: str = None,
    ) -> Dict[int, dict]:
        """Test Opus robustness using the primary model for evaluation."""
        # Create a lightweight single-model attacker for robustness testing
        # (reuses self.codec and self.target_model)
        helper = object.__new__(LatentCodecAttacker)
        helper.codec = self.codec
        helper.target_model = self.target_model
        helper.device = self.device
        helper.verbose = self.verbose
        helper.log = self.log
        return helper.test_opus_robustness(result, bitrates=bitrates, prompt=prompt)

    def test_encodec_robustness(
        self,
        result: AttackResult,
        bitrates: List[float] = None,
        prompt: str = None,
    ) -> Dict[float, dict]:
        """Test EnCodec re-encoding robustness using the primary model."""
        helper = object.__new__(LatentCodecAttacker)
        helper.codec = self.codec
        helper.target_model = self.target_model
        helper.device = self.device
        helper.verbose = self.verbose
        helper.log = self.log
        helper._bandwidth = getattr(self.codec, '_bandwidth', ENCODEC_BANDWIDTH)
        return helper.test_encodec_robustness(result, bitrates=bitrates, prompt=prompt)

    def save_adversarial_wav(self, result: AttackResult, audio_dir: str,
                             filename: str) -> str:
        """Save adversarial WAV to a shared audio directory."""
        os.makedirs(audio_dir, exist_ok=True)
        wav_path = os.path.join(audio_dir, f"{filename}.wav")
        if os.path.isfile(wav_path):
            self.log(f"Audio already exists, skipping: {wav_path}")
            return wav_path
        import soundfile as sf
        adv_wav = result.adversarial_wav.squeeze().cpu().numpy()
        sf.write(wav_path, adv_wav, ENCODEC_SAMPLE_RATE)
        self.log(f"Saved adversarial audio: {wav_path}")
        return wav_path
