"""Verify Qwen2-Audio wav_to_mel() still matches processor after fix."""
import sys
sys.path.insert(0, '${PROJECT_ROOT}/external/whisper-inject-v2-fork')

import torch
import numpy as np

from models.qwen2_audio import Qwen2AudioModel
model = Qwen2AudioModel(device="cuda", dtype=torch.bfloat16)

import librosa
wav_np, sr = librosa.load('${PROJECT_ROOT}/data/music/empire_state_of_mind.mp3', sr=16000, duration=10.0)
wav = torch.from_numpy(wav_np).unsqueeze(0).to("cuda")

# Our mel
our_mel = model.wav_to_mel(wav)

# Processor mel
conversation = [{"role": "user", "content": [{"type": "audio", "audio_url": "placeholder"}, {"type": "text", "text": "test"}]}]
text = model.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
inputs = model.processor(text=text, audio=[wav_np], return_tensors="pt", sampling_rate=16000)
proc_mel = inputs['input_features']

our_f32 = our_mel.float().cpu()
proc_f32 = proc_mel.float().cpu()
min_len = min(our_f32.shape[2], proc_f32.shape[2])
diff = (our_f32[:,:,:min_len] - proc_f32[:,:,:min_len]).abs()

print(f"Qwen2 mel shapes: ours={our_mel.shape}, proc={proc_mel.shape}")
print(f"Mean diff: {diff.mean().item():.6f}")
print(f"Max diff: {diff.max().item():.6f}")

if diff.mean().item() < 0.01:
    print("SUCCESS: Qwen2 mel features match processor!")
else:
    print("WARNING: Qwen2 mel features differ!")
