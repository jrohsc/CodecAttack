"""One-off DAC API discovery probe. Delete after task 1 confirms the wrapper interface."""
import torch
import dac

# 1. Load a 24kHz model
model_path = dac.utils.download(model_type="24khz")
print(f"\n=== model path ===\n{model_path}")

model = dac.DAC.load(model_path)
model = model.to("cuda")
model.train(False)  # inference mode
print(f"\n=== model class ===\n{type(model).__name__}")
print(f"\n=== top-level attrs ===")
for a in sorted(dir(model)):
    if a.startswith("_"):
        continue
    v = getattr(model, a, None)
    cls = type(v).__name__ if v is not None else "None"
    print(f"  {a:30s} :: {cls}")

# 2. Round-trip a 1s sine to discover encode/decode call signatures
sr = 24000
audio = torch.randn(1, 1, sr).to("cuda") * 0.1
print(f"\n=== input audio shape: {audio.shape} ===")

# Try high-level encode (likely returns multiple tensors)
try:
    out = model.encode(audio)
    print(f"\n=== model.encode(audio) returned: {type(out).__name__} ===")
    if isinstance(out, tuple):
        for i, t in enumerate(out):
            shape = t.shape if hasattr(t, "shape") else "(scalar/None)"
            print(f"  [{i}] {type(t).__name__} shape={shape}")
    elif isinstance(out, dict):
        for k, t in out.items():
            shape = t.shape if hasattr(t, "shape") else "(scalar/None)"
            print(f"  [{k!r}] {type(t).__name__} shape={shape}")
except Exception as e:
    print(f"!!! model.encode failed: {e}")

# Try direct encoder (the EnCodec-style path we want)
try:
    feats = model.encoder(audio)
    print(f"\n=== model.encoder(audio).shape = {feats.shape} ===")
except Exception as e:
    print(f"!!! model.encoder failed: {e}")

# Probe quantizer for in_proj / out_proj projector access
print(f"\n=== quantizer attrs ===")
q = getattr(model, "quantizer", None)
if q is not None:
    for a in sorted(dir(q)):
        if a.startswith("_"):
            continue
        print(f"  {a}")

# Try decode from continuous z
if "feats" in dir():
    try:
        rec = model.decoder(feats)
        print(f"\n=== model.decoder(feats).shape = {rec.shape} ===")
    except Exception as e:
        print(f"!!! model.decoder(feats) failed: {e}")
