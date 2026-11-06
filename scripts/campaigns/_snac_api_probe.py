"""
SNAC API probe: verify continuous-bypass attack hooks before committing to a wrapper.

Checks:
  1. encoder(audio) output type and shape
  2. encoder -> decoder roundtrip MSE (no quantization)
  3. encoder -> quantizer -> decoder roundtrip MSE (with quantization, for reference)
  4. Gradient flow: dL/dz where L = MSE(decoder(z) - audio)
  5. Latent magnitude statistics (for setting epsilon scale)
"""
import torch
from snac import SNAC

DEVICE = "cuda"
SR = 24000
DUR = 4.0  # seconds


def main() -> None:
    print("[probe] loading SNAC 24kHz...")
    model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to(DEVICE)
    print(f"[probe] sampling_rate={model.sampling_rate}  hop_length={model.hop_length}")
    print(f"[probe] latent_dim={model.latent_dim}  n_codebooks={model.n_codebooks}  vq_strides={model.vq_strides}")

    # synthetic 24kHz mono carrier
    n = int(SR * DUR)
    torch.manual_seed(0)
    audio = torch.randn(1, 1, n, device=DEVICE) * 0.1
    print(f"[probe] input audio shape {tuple(audio.shape)}")

    # 1+5: encoder output
    audio_pad = model.preprocess(audio)
    print(f"[probe] preprocessed audio shape {tuple(audio_pad.shape)} (after pad)")
    with torch.no_grad():
        z = model.encoder(audio_pad)
    print(f"[probe] encoder(z) type={type(z).__name__}  shape={tuple(z.shape)}")
    print(f"[probe] z stats: mean={z.mean().item():.4f}  std={z.std().item():.4f}  abs_max={z.abs().max().item():.4f}")

    # 2: continuous roundtrip (no VQ)
    with torch.no_grad():
        recon_continuous = model.decoder(z)
    mse_cont = torch.mean((recon_continuous[..., :n] - audio) ** 2).item()
    print(f"[probe] continuous roundtrip MSE = {mse_cont:.6e}  (lower is better)")

    # 3: full roundtrip (with VQ)
    with torch.no_grad():
        z_q, codes = model.quantizer(z)
        recon_vq = model.decoder(z_q)
    mse_vq = torch.mean((recon_vq[..., :n] - audio) ** 2).item()
    print(f"[probe] VQ roundtrip MSE = {mse_vq:.6e}  (codes: {[tuple(c.shape) for c in codes]})")

    # 4: gradient flow through decoder
    model.decoder.train()  # enable grad path through any train-only layers
    z_var = z.detach().clone().requires_grad_(True)
    audio_recon = model.decoder(z_var)
    loss = torch.mean(audio_recon ** 2)
    loss.backward()
    grad_norm = z_var.grad.norm().item()
    print(f"[probe] dL/dz norm = {grad_norm:.4f}  (nonzero => grads flow)")

    # 5: temporal stride check
    t_z = z.shape[-1]
    t_audio = audio_pad.shape[-1]
    print(f"[probe] z time = {t_z}  audio time = {t_audio}  stride = {t_audio / t_z:.1f}")
    print(f"[probe] z frame rate = {SR / (t_audio / t_z):.2f} Hz")

    print("[probe] DONE")


if __name__ == "__main__":
    main()
