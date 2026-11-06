"""Quick test: verify gradient flow through soft-RVQ for PersonaPLEX attack."""
import sys
import os

# Setup paths (same as latent_attack.py)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
QUALCOMM_ROOT = os.path.dirname(PROJECT_ROOT)
WHISPER_INJECT_ROOT = os.path.join(QUALCOMM_ROOT, "external", "whisper-inject-v2-fork")
sys.path.insert(0, WHISPER_INJECT_ROOT)

import torch

print("=" * 60)
print("PersonaPLEX Soft-RVQ Gradient Flow Test")
print("=" * 60)

# Step 1: Load model
print("\n[1/5] Loading PersonaPLEX/Moshi model...")
from models.personaplex import PersonaPLEXModel
model = PersonaPLEXModel(device="cuda", dtype=torch.bfloat16)

# Step 2: Create codec wrapper
print("\n[2/5] Creating MimiCodecWrapper from loaded model...")
from attacks.latent_codec_mimi import MimiCodecWrapper
codec = MimiCodecWrapper(model=model.mimi, device="cuda", dtype=torch.float32)
print(f"  Codec dims: hidden={codec.hidden_size}, vq_dim={codec.vq_dim}, "
      f"codebook_size={codec.codebook_size}")

# Step 3: Encode test audio
print("\n[3/5] Encoding test audio (1s random noise)...")
test_wav = torch.randn(1, 1, 24000, device="cuda", dtype=torch.float32) * 0.01
z = codec.encode_to_continuous(test_wav)
print(f"  Encoded z: {z.shape}, dtype={z.dtype}")

# Step 4: Test gradient flow
print("\n[4/5] Testing gradient flow: delta -> soft-RVQ -> Moshi decoder -> CE loss...")
delta = torch.zeros_like(z, requires_grad=True)
z_adv = z.detach() + delta

target_text = "Turn left at the next intersection"
loss = model.compute_loss_from_latents(z_adv, target_text, temperature=0.1)
print(f"  Loss value: {loss.item():.4f}")
print(f"  Loss has grad_fn: {loss.grad_fn is not None}")

loss.backward()
print(f"  delta.grad exists: {delta.grad is not None}")
if delta.grad is not None:
    grad_norm = delta.grad.norm().item()
    nonzero = (delta.grad != 0).sum().item()
    total = delta.grad.numel()
    print(f"  delta.grad norm: {grad_norm:.6f}")
    print(f"  delta.grad nonzero: {nonzero} / {total} ({100*nonzero/total:.1f}%)")

# Step 5: Verify decode path works
print("\n[5/5] Testing decode path (for audio output)...")
codec.set_decode_train_mode(True)
z_adv2 = z.detach() + delta.detach()
audio_out = codec.decode_from_continuous(z_adv2)
print(f"  Decoded audio: {audio_out.shape}, dtype={audio_out.dtype}")
codec.set_decode_train_mode(False)

# Summary
print("\n" + "=" * 60)
if delta.grad is not None and grad_norm > 0:
    print("RESULT: GRADIENT FLOW TEST PASSED")
else:
    print("RESULT: GRADIENT FLOW TEST FAILED")
print("=" * 60)
