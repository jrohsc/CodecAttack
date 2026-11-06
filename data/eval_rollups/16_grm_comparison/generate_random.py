import argparse
import numpy as np
import torch


def rms(x: torch.Tensor, eps: float = 1e-12) -> float:
    return float(torch.sqrt(torch.mean(x.float() ** 2) + eps).item())


def masked_rms(tile: torch.Tensor, mask_1xFx1: torch.Tensor, eps: float = 1e-12) -> float:
    # Only consider masked (non-zero) bins for RMS to match perturbation budget fairly.
    x = (tile * mask_1xFx1).float()
    sel = mask_1xFx1.expand_as(x) > 0
    if not bool(sel.any()):
        return 0.0
    return float(torch.sqrt(torch.mean(x[sel] ** 2) + eps).item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mask_npz", default="./logs/48_noise/top48_weighted.npz")
    ap.add_argument("--best_pt", default="./logs/48_noise/checkpoints/best.pt")
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iters", type=int, default=5, help="Number of clamp-rescale iterations to better match masked RMS")
    ap.add_argument("--save_pt", type=str, default="./logs/48_noise/checkpoints/random.pt")
    args = ap.parse_args()

    # Load mask
    m = np.load(args.mask_npz)
    if "mask" not in m.files:
        raise KeyError(f"'mask' not found in {args.mask_npz}. keys={m.files}")
    mask = torch.from_numpy(np.asarray(m["mask"], dtype=np.float32)).view(1, -1, 1)

    # Load tile
    obj = torch.load(args.best_pt, map_location="cpu")
    tile = obj["tile"] if isinstance(obj, dict) and "tile" in obj else obj
    if not isinstance(tile, torch.Tensor):
        raise ValueError(f"Unsupported best.pt format: {type(obj)}")

    print("== Loaded ==")
    print(f"mask_npz: {args.mask_npz}")
    print(f"mask shape: {tuple(mask.shape)} dtype: {mask.dtype} ones: {int(mask.sum().item())}/{mask.numel()}")
    print(f"best_pt: {args.best_pt}")
    print(f"tile shape: {tuple(tile.shape)} dtype: {tile.dtype} min: {float(tile.min())} max: {float(tile.max())}")

    # Clamp to tau (should already be within [-tau, tau])
    tile_c = tile.clamp(min=-args.tau, max=args.tau)

    print("== Energy stats ==")
    print(f"tile RMS (all bins): {rms(tile_c):.6f}")
    print(f"tile RMS (masked bins): {masked_rms(tile_c, mask):.6f}")

    # Build random baseline tile with matched masked RMS
    torch.manual_seed(args.seed)
    rand = torch.randn_like(tile_c)
    rand = rand * mask  # enforce same band constraint

    best_r = masked_rms(tile_c, mask)

    # Iterative clamp-rescale to better match the post-clamp masked RMS
    scale = 1.0
    for it in range(max(1, int(args.iters))):
        rand_r = masked_rms(rand, mask)
        step_scale = 0.0 if rand_r == 0 else (best_r / rand_r)
        scale *= step_scale
        rand = (rand * step_scale).clamp(min=-args.tau, max=args.tau)

    print("== Random (masked, RMS-matched; iterative clamp-rescale) ==")
    print(f"seed: {args.seed}")
    print(f"iters: {int(args.iters)}")
    print(f"total scale factor: {scale:.6f}")
    print(f"rand min/max: {float(rand.min())} / {float(rand.max())}")
    print(f"rand RMS (masked bins): {masked_rms(rand, mask):.6f}")
    print(f"rand RMS (all bins): {rms(rand):.6f}")

    print("== Diagnostics ==")
    diff = abs(masked_rms(rand, mask) - best_r)
    print(f"|rand_masked_rms - best_masked_rms| = {diff:.6f}")

    if args.save_pt:
        payload = {"tile": rand.detach().cpu(), "meta": {"seed": int(args.seed), "tau": float(args.tau), "iters": int(args.iters), "best_masked_rms": float(best_r)}}
        torch.save(payload, args.save_pt)
        print(f"Saved random tile -> {args.save_pt}")


if __name__ == "__main__":
    main()

