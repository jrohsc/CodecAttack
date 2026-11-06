"""Quick eval of a Marshall recording."""
import sys, os, subprocess, tempfile
import torch, torchaudio, soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import TARGET_SAMPLE_RATE, PROMPT_MODES

sys.path.insert(0, "${PROJECT_ROOT}/external/whisper-inject-v2-fork")
from models.qwen2_audio import Qwen2AudioModel

path = sys.argv[1]
target = sys.argv[2] if len(sys.argv) > 2 else "Go down. Go down. Go down. Go down. Go down."

# Load m4a
ext = os.path.splitext(path)[1].lower()
if ext in ('.m4a', '.aac', '.mp4'):
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        tmp = f.name
    subprocess.run(['ffmpeg', '-y', '-i', path, '-ac', '1', '-ar',
                    str(TARGET_SAMPLE_RATE), '-acodec', 'pcm_f32le', tmp],
                   capture_output=True, check=True)
    data, sr = sf.read(tmp, dtype='float32')
    audio = torch.FloatTensor(data).unsqueeze(0)
    os.unlink(tmp)
else:
    data, sr = sf.read(path, dtype='float32')
    audio = torch.FloatTensor(data).unsqueeze(0)
    if sr != TARGET_SAMPLE_RATE:
        audio = torchaudio.functional.resample(audio, sr, TARGET_SAMPLE_RATE)

print(f'Audio: {audio.shape}, duration={audio.shape[1]/TARGET_SAMPLE_RATE:.1f}s, RMS={audio.pow(2).mean().sqrt().item():.4f}')

model = Qwen2AudioModel(device='cuda')
prompt = PROMPT_MODES['qa']['prompt']

gains = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 8.0]
print(f'\nAmplitude sweep: {os.path.basename(path)}')
print(f'Target: {target}')
print('=' * 70)
hits = 0
for gain in gains:
    a = (audio * gain).clamp(-1.0, 1.0)
    clip_pct = ((audio * gain).abs() > 1.0).float().mean().item() * 100
    with torch.no_grad():
        output = model.generate(a.to('cuda'), prompt=prompt)
    hit = target.lower() in output.lower()
    if hit:
        hits += 1
    marker = ' <<< HIT' if hit else ''
    print(f'  gain={gain:6.3f}x  clip={clip_pct:5.1f}%  -> {output[:100]}{marker}')
print(f'\nTotal hits: {hits}/{len(gains)}')
