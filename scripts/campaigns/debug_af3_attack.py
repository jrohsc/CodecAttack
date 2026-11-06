"""
Diagnostic script: debug why Audio Flamingo 3 attack loss doesn't converge.

Checks:
1. AF3 audio encoder architecture vs Qwen2-Audio
2. Gradient flow: does loss have grad_fn all the way back to delta?
3. Gradient magnitudes at each stage of the pipeline
4. Loss curve over first 50 steps
5. Input features mask correctness
6. Whether AF3's internal audio encoder is frozen / non-differentiable

Run in flamingo3 conda env:
    conda activate flamingo3
    python debug_af3_attack.py
"""

import sys
import os
import torch
import torchaudio
import numpy as np
import time

# Setup paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
QUALCOMM_ROOT = os.path.dirname(PROJECT_ROOT)
WHISPER_INJECT_ROOT = os.path.join(QUALCOMM_ROOT, "external", "whisper-inject-v2-fork")
sys.path.insert(0, WHISPER_INJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

from config import MODEL_PATHS, ENCODEC_SAMPLE_RATE, TARGET_SAMPLE_RATE
from music_carrier import load_music_by_name


def check_model_architecture(model_wrapper):
    """Print audio encoder architecture details."""
    print("\n" + "=" * 70)
    print("1. MODEL ARCHITECTURE")
    print("=" * 70)

    model = model_wrapper.model

    # Print top-level modules
    print("\nTop-level modules:")
    for name, module in model.named_children():
        param_count = sum(p.numel() for p in module.parameters())
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        print(f"  {name}: {param_count:,} params ({trainable:,} trainable)")

    # Find the audio encoder specifically
    print("\nSearching for audio encoder...")
    for name, module in model.named_modules():
        name_lower = name.lower()
        if any(k in name_lower for k in ['audio_encoder', 'audio_tower', 'whisper', 'audio_model']):
            if '.' not in name or name.count('.') <= 1:
                param_count = sum(p.numel() for p in module.parameters())
                requires_grad = any(p.requires_grad for p in module.parameters())
                print(f"  Found: {name} ({type(module).__name__})")
                print(f"    params: {param_count:,}, requires_grad: {requires_grad}")

    # Check model config for audio encoder type
    if hasattr(model, 'config'):
        config = model.config
        print(f"\nModel config type: {type(config).__name__}")
        for attr in ['audio_config', 'audio_encoder_config', 'encoder_config',
                     'audio_model_type', 'audio_encoder_type']:
            if hasattr(config, attr):
                val = getattr(config, attr)
                print(f"  {attr}: {val}")

        # Print all config attributes that mention 'audio'
        for attr in dir(config):
            if 'audio' in attr.lower() and not attr.startswith('_'):
                val = getattr(config, attr)
                if not callable(val):
                    print(f"  {attr}: {val}")


def check_gradient_flow(model_wrapper, codec, music_wav, target_text, prompt):
    """Run one forward+backward pass and check gradient health."""
    print("\n" + "=" * 70)
    print("2. GRADIENT FLOW CHECK")
    print("=" * 70)

    device = model_wrapper.device

    # Encode music
    music_wav = music_wav.to(device)
    with torch.no_grad():
        z_original = codec.encode_to_continuous(music_wav)
    print(f"z_original shape: {z_original.shape}, dtype: {z_original.dtype}")

    # Create delta
    delta = torch.zeros_like(z_original, requires_grad=True)

    # Enable decoder gradients
    codec.model.decoder.train()

    # Forward pass
    z_adv = z_original + delta
    print(f"\nz_adv requires_grad: {z_adv.requires_grad}")

    audio_24k = codec.decode_from_continuous(z_adv)
    print(f"audio_24k requires_grad: {audio_24k.requires_grad}, grad_fn: {audio_24k.grad_fn}")

    audio_16k = torchaudio.functional.resample(
        audio_24k.squeeze(0), ENCODEC_SAMPLE_RATE, TARGET_SAMPLE_RATE
    )
    print(f"audio_16k requires_grad: {audio_16k.requires_grad}, shape: {audio_16k.shape}")

    # Compute mel
    mel = model_wrapper.wav_to_mel(audio_16k)
    print(f"mel requires_grad: {mel.requires_grad}, grad_fn: {mel.grad_fn}")
    print(f"mel shape: {mel.shape}, dtype: {mel.dtype}")

    input_features = mel.to(model_wrapper.dtype)
    print(f"input_features requires_grad: {input_features.requires_grad}, dtype: {input_features.dtype}")

    # Check if grad_fn chain is intact
    print("\nGrad function chain (mel -> ... -> delta):")
    fn = mel.grad_fn
    depth = 0
    while fn is not None and depth < 20:
        print(f"  [{depth}] {type(fn).__name__}")
        fn = fn.next_functions[0][0] if fn.next_functions else None
        depth += 1

    # Compute loss
    print(f"\nComputing loss (target: '{target_text[:50]}...')")
    print(f"  prompt: '{prompt}'")

    loss = model_wrapper.compute_loss(audio_16k, target_text, prompt=prompt)
    print(f"\nloss value: {loss.item():.6f}")
    print(f"loss requires_grad: {loss.requires_grad}")
    print(f"loss grad_fn: {loss.grad_fn}")

    if not loss.requires_grad:
        print("\n*** CRITICAL: loss does NOT require grad! Gradient chain is broken. ***")
        codec.model.decoder.eval()
        return

    # Backward
    loss.backward()

    if delta.grad is None:
        print("\n*** CRITICAL: delta.grad is None after backward! ***")
        print("Gradient chain is broken somewhere between loss and delta.")
    else:
        grad_norm = delta.grad.norm().item()
        grad_max = delta.grad.abs().max().item()
        grad_mean = delta.grad.abs().mean().item()
        grad_nonzero = (delta.grad.abs() > 1e-10).sum().item()
        total_elements = delta.grad.numel()
        print(f"\ndelta.grad stats:")
        print(f"  norm:     {grad_norm:.6e}")
        print(f"  max:      {grad_max:.6e}")
        print(f"  mean:     {grad_mean:.6e}")
        print(f"  nonzero:  {grad_nonzero}/{total_elements} ({100*grad_nonzero/total_elements:.1f}%)")

        if grad_norm < 1e-8:
            print("\n*** WARNING: Gradient norm is near zero! Vanishing gradients. ***")
        elif grad_norm > 1e4:
            print("\n*** WARNING: Gradient norm is very large! Exploding gradients. ***")

    codec.model.decoder.eval()


def check_loss_convergence(model_wrapper, codec, music_wav, target_text, prompt,
                           steps=50, lr=0.2, eps=10.0):
    """Run N optimization steps and track loss + gradient norms."""
    print("\n" + "=" * 70)
    print(f"3. LOSS CONVERGENCE ({steps} steps, lr={lr}, eps={eps})")
    print("=" * 70)

    device = model_wrapper.device

    music_wav = music_wav.to(device)
    with torch.no_grad():
        z_original = codec.encode_to_continuous(music_wav)
        original_audio_24k = codec.decode_from_continuous(z_original)

    delta = torch.zeros_like(z_original, requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=lr)

    codec.model.decoder.train()

    losses = []
    grad_norms = []
    delta_norms = []

    for step in range(steps):
        optimizer.zero_grad()

        z_adv = z_original + delta
        audio_24k = codec.decode_from_continuous(z_adv)
        audio_16k = torchaudio.functional.resample(
            audio_24k.squeeze(0), ENCODEC_SAMPLE_RATE, TARGET_SAMPLE_RATE
        )

        loss = model_wrapper.compute_loss(audio_16k, target_text, prompt=prompt)
        loss.backward()

        # Record metrics before optimizer step
        loss_val = loss.item()
        grad_norm = delta.grad.norm().item() if delta.grad is not None else 0.0
        delta_norm = delta.data.norm().item()

        losses.append(loss_val)
        grad_norms.append(grad_norm)
        delta_norms.append(delta_norm)

        optimizer.step()

        with torch.no_grad():
            delta.data = torch.clamp(delta.data, -eps, eps)

        if (step + 1) % 5 == 0 or step == 0:
            print(f"  Step {step+1:3d} | loss: {loss_val:.6f} | "
                  f"grad_norm: {grad_norm:.4e} | delta_norm: {delta_norm:.4f}")

    codec.model.decoder.eval()

    # Analysis
    print(f"\nLoss trajectory:")
    print(f"  Start:    {losses[0]:.6f}")
    print(f"  End:      {losses[-1]:.6f}")
    print(f"  Min:      {min(losses):.6f} (step {losses.index(min(losses))+1})")
    print(f"  Decrease: {losses[0] - losses[-1]:.6f} ({100*(losses[0]-losses[-1])/losses[0]:.1f}%)")

    print(f"\nGradient norms:")
    print(f"  Start:    {grad_norms[0]:.4e}")
    print(f"  End:      {grad_norms[-1]:.4e}")
    print(f"  Mean:     {np.mean(grad_norms):.4e}")
    print(f"  Std:      {np.std(grad_norms):.4e}")

    if losses[-1] > 0.9 * losses[0]:
        print("\n*** Loss barely decreased! Possible issues:")
        if np.mean(grad_norms) < 1e-6:
            print("  -> Vanishing gradients (grad_norm too small)")
        elif np.mean(grad_norms) > 1e3:
            print("  -> Exploding gradients (grad_norm too large)")
        else:
            print("  -> Gradients exist but optimization is stuck")
            print("  -> Try different lr, more steps, or check loss surface")

    # Check output at final step
    with torch.no_grad():
        z_test = z_original + delta
        audio_test_24k = codec.decode_from_continuous(z_test)
        audio_test_16k = torchaudio.functional.resample(
            audio_test_24k.squeeze(0), ENCODEC_SAMPLE_RATE, TARGET_SAMPLE_RATE
        )
        output = model_wrapper.generate(audio_test_16k, prompt=prompt)
    print(f"\nFinal model output: {output[:200]}")
    print(f"Target text:        {target_text}")

    return losses, grad_norms


def check_lr_sensitivity(model_wrapper, codec, music_wav, target_text, prompt,
                         eps=10.0):
    """Test multiple learning rates to find the best one."""
    print("\n" + "=" * 70)
    print("4. LEARNING RATE SENSITIVITY")
    print("=" * 70)

    device = model_wrapper.device
    music_wav = music_wav.to(device)

    with torch.no_grad():
        z_original = codec.encode_to_continuous(music_wav)

    lrs = [0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
    test_steps = 30

    for lr in lrs:
        delta = torch.zeros_like(z_original, requires_grad=True)
        optimizer = torch.optim.Adam([delta], lr=lr)
        codec.model.decoder.train()

        start_loss = None
        end_loss = None

        for step in range(test_steps):
            optimizer.zero_grad()
            z_adv = z_original + delta
            audio_24k = codec.decode_from_continuous(z_adv)
            audio_16k = torchaudio.functional.resample(
                audio_24k.squeeze(0), ENCODEC_SAMPLE_RATE, TARGET_SAMPLE_RATE
            )
            loss = model_wrapper.compute_loss(audio_16k, target_text, prompt=prompt)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                delta.data = torch.clamp(delta.data, -eps, eps)

            if step == 0:
                start_loss = loss.item()
            end_loss = loss.item()

        codec.model.decoder.eval()
        decrease = start_loss - end_loss
        pct = 100 * decrease / start_loss if start_loss > 0 else 0
        print(f"  lr={lr:<6} | start={start_loss:.4f} end={end_loss:.4f} | "
              f"decrease={decrease:.4f} ({pct:.1f}%)")


def check_input_features_mask(model_wrapper, wav):
    """Check if AF3's input_features_mask is computed correctly."""
    print("\n" + "=" * 70)
    print("5. INPUT FEATURES MASK CHECK")
    print("=" * 70)

    wav_np = wav.detach().squeeze().cpu().numpy()
    conversation = [{
        "role": "user",
        "content": [
            {"type": "audio", "audio": wav_np},
            {"type": "text", "text": "What action is being requested in this audio?"}
        ]
    }]
    text = model_wrapper.processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
    )
    inputs = model_wrapper.processor(
        text=text, audio=[wav_np],
        return_tensors="pt", sampling_rate=model_wrapper.SAMPLE_RATE
    )

    print(f"Processor output keys: {list(inputs.keys())}")
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: shape={v.shape}, dtype={v.dtype}")

    if 'input_features' in inputs:
        feat = inputs['input_features']
        print(f"\ninput_features from processor:")
        print(f"  shape: {feat.shape}")
        print(f"  min: {feat.min():.4f}, max: {feat.max():.4f}, mean: {feat.mean():.4f}")

    if 'input_features_mask' in inputs:
        mask = inputs['input_features_mask']
        print(f"\ninput_features_mask:")
        print(f"  shape: {mask.shape}")
        print(f"  sum (valid frames): {mask.sum().item()}")
        print(f"  total frames: {mask.numel()}")
    elif 'feature_attention_mask' in inputs:
        mask = inputs['feature_attention_mask']
        print(f"\nfeature_attention_mask:")
        print(f"  shape: {mask.shape}")
        print(f"  sum (valid frames): {mask.sum().item()}")
        print(f"  total frames: {mask.numel()}")
    else:
        print("\n*** WARNING: No feature mask found in processor output! ***")

    # Compare processor mel with our differentiable mel
    our_mel = model_wrapper.wav_to_mel(wav)
    print(f"\nOur differentiable mel:")
    print(f"  shape: {our_mel.shape}")
    print(f"  min: {our_mel.min():.4f}, max: {our_mel.max():.4f}, mean: {our_mel.mean():.4f}")

    if 'input_features' in inputs:
        proc_mel = inputs['input_features']
        if our_mel.shape == proc_mel.shape:
            diff = (our_mel.float().cpu() - proc_mel.float().cpu()).abs()
            print(f"\nMel difference (ours vs processor):")
            print(f"  max diff: {diff.max():.6f}")
            print(f"  mean diff: {diff.mean():.6f}")
            if diff.max() > 0.01:
                print("  *** WARNING: Mel features don't match processor! ***")
            else:
                print("  OK: Mel features match processor closely.")
        else:
            print(f"\n*** WARNING: Shape mismatch! ours={our_mel.shape} vs proc={proc_mel.shape} ***")


def compare_with_qwen2(codec, music_wav, target_text, prompt):
    """Try loading Qwen2 for comparison (may fail if wrong transformers version)."""
    print("\n" + "=" * 70)
    print("6. COMPARISON WITH QWEN2-AUDIO (if available)")
    print("=" * 70)

    try:
        from models.qwen2_audio import Qwen2AudioModel
        qwen2 = Qwen2AudioModel(device="cuda", dtype=torch.bfloat16)
    except Exception as e:
        print(f"Cannot load Qwen2-Audio in this env: {e}")
        print("(Expected - Qwen2 needs transformers 4.x, AF3 needs 5.x)")
        return

    device = qwen2.device
    music_wav = music_wav.to(device)

    with torch.no_grad():
        z_original = codec.encode_to_continuous(music_wav)

    delta = torch.zeros_like(z_original, requires_grad=True)
    codec.model.decoder.train()

    z_adv = z_original + delta
    audio_24k = codec.decode_from_continuous(z_adv)
    audio_16k = torchaudio.functional.resample(
        audio_24k.squeeze(0), ENCODEC_SAMPLE_RATE, TARGET_SAMPLE_RATE
    )

    loss = qwen2.compute_loss(audio_16k, target_text, prompt=prompt)
    loss.backward()

    print(f"Qwen2 loss: {loss.item():.6f}")
    if delta.grad is not None:
        print(f"Qwen2 delta.grad norm: {delta.grad.norm().item():.4e}")
    else:
        print("Qwen2 delta.grad: None")

    codec.model.decoder.eval()
    del qwen2
    torch.cuda.empty_cache()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Debug AF3 attack convergence")
    parser.add_argument("--music", type=str, default="jazz_1",
                        help="Music carrier to test with")
    parser.add_argument("--target", type=str,
                        default="Turn left at the next intersection",
                        help="Target text")
    parser.add_argument("--prompt", type=str,
                        default="What action is being requested in this audio?",
                        help="Model prompt")
    parser.add_argument("--eps", type=float, default=10.0)
    parser.add_argument("--steps", type=int, default=50,
                        help="Steps for convergence test")
    parser.add_argument("--skip-compare", action="store_true",
                        help="Skip Qwen2 comparison (different env)")
    args = parser.parse_args()

    print("=" * 70)
    print("AF3 ATTACK CONVERGENCE DIAGNOSTIC")
    print("=" * 70)

    import transformers
    print(f"transformers version: {transformers.__version__}")
    print(f"torch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        props = torch.cuda.get_device_properties(0)
        total_mem = getattr(props, 'total_memory', getattr(props, 'total_mem', 0))
        print(f"GPU memory: {total_mem / 1e9:.1f} GB")

    # Load EnCodec
    print("\nLoading EnCodec...")
    from attacks.latent_codec import EnCodecWrapper
    codec = EnCodecWrapper(bandwidth=6.0, device="cuda")

    # Load music
    print(f"\nLoading music: {args.music}")
    music_wav = load_music_by_name(args.music)
    print(f"Music shape: {music_wav.shape}, duration: {music_wav.shape[-1]/ENCODEC_SAMPLE_RATE:.1f}s")

    # Load AF3
    print("\nLoading Audio Flamingo 3...")
    from models.audio_flamingo import AudioFlamingoModel
    af3 = AudioFlamingoModel(device="cuda", dtype=torch.bfloat16)

    # Prepare 16kHz audio for mask check
    music_wav_device = music_wav.to("cuda")
    with torch.no_grad():
        z = codec.encode_to_continuous(music_wav_device)
        audio_24k = codec.decode_from_continuous(z)
        audio_16k = torchaudio.functional.resample(
            audio_24k.squeeze(0), ENCODEC_SAMPLE_RATE, TARGET_SAMPLE_RATE
        )

    # Run diagnostics
    check_model_architecture(af3)
    check_input_features_mask(af3, audio_16k)
    check_gradient_flow(af3, codec, music_wav, args.target, args.prompt)
    check_loss_convergence(af3, codec, music_wav, args.target, args.prompt,
                           steps=args.steps, lr=0.2, eps=args.eps)
    check_lr_sensitivity(af3, codec, music_wav, args.target, args.prompt,
                         eps=args.eps)

    if not args.skip_compare:
        compare_with_qwen2(codec, music_wav, args.target, args.prompt)

    print("\n" + "=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
