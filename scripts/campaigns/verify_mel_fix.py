"""Verify that wav_to_mel() matches processor mel features after the fix."""
import sys
sys.path.insert(0, '${PROJECT_ROOT}/external/whisper-inject-v2-fork')

import torch
import numpy as np

# Load AF3 model
from models.audio_flamingo import AudioFlamingoModel
model = AudioFlamingoModel(device="cuda", dtype=torch.bfloat16)

# Load a test audio
import librosa
music_path = '${PROJECT_ROOT}/data/music/empire_state_of_mind.mp3'
wav_np, sr = librosa.load(music_path, sr=16000, duration=10.0)
wav = torch.from_numpy(wav_np).unsqueeze(0).to("cuda")

# 1. Our wav_to_mel
our_mel = model.wav_to_mel(wav)
print(f"Our mel shape: {our_mel.shape}")
print(f"Our mel range: [{our_mel.min().item():.4f}, {our_mel.max().item():.4f}]")

# 2. Processor mel
proc_inputs = model.processor(
    text="test", audio=[wav_np],
    return_tensors="pt", sampling_rate=16000
)
proc_mel = proc_inputs['input_features']
print(f"\nProcessor mel shape: {proc_mel.shape}")
print(f"Processor mel range: [{proc_mel.min().item():.4f}, {proc_mel.max().item():.4f}]")

# 3. Compare
our_mel_f32 = our_mel.float().cpu()
proc_mel_f32 = proc_mel.float().cpu()

# Match shapes
min_len = min(our_mel_f32.shape[2], proc_mel_f32.shape[2])
diff = (our_mel_f32[:, :, :min_len] - proc_mel_f32[:, :, :min_len]).abs()
print(f"\nMean absolute difference: {diff.mean().item():.6f}")
print(f"Max absolute difference: {diff.max().item():.6f}")
print(f"Relative error: {(diff / (proc_mel_f32[:, :, :min_len].abs() + 1e-8)).mean().item():.6f}")

# Check audio region vs padded region separately
# 10s audio = 1000 mel frames, rest is padding
audio_frames = int(10.0 * 16000 / 160)  # 1000
print(f"\nAudio region (0:{audio_frames}):")
audio_diff = diff[:, :, :audio_frames]
print(f"  Mean diff: {audio_diff.mean().item():.6f}")
print(f"  Max diff: {audio_diff.max().item():.6f}")

print(f"\nPadded region ({audio_frames}:{min_len}):")
pad_diff = diff[:, :, audio_frames:min_len]
print(f"  Mean diff: {pad_diff.mean().item():.6f}")
print(f"  Max diff: {pad_diff.max().item():.6f}")

# Sample values comparison in padded region
print(f"\nOur mel padded sample [0,:5,{audio_frames}]: {our_mel_f32[0,:5,audio_frames].tolist()}")
print(f"Proc mel padded sample [0,:5,{audio_frames}]: {proc_mel_f32[0,:5,audio_frames].tolist()}")

print("\n=== VERIFICATION COMPLETE ===")
if diff.mean().item() < 0.01:
    print("SUCCESS: Mel features match processor within tolerance!")
else:
    print("WARNING: Mel features still differ significantly.")
