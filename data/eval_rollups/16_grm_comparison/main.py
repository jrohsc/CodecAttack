import argparse
import math
import os
import time
from typing import List, Iterable, Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from dataloader import AudioDataset, collate_fn

# Optional experiment tracking
try:
    import swanlab
except Exception:
    swanlab = None

from gradient import (
    load_model_and_processor,
    build_chat_texts,
    build_labels,
    build_labels_suffix,
    _detect_feature_key,
    call_processor,
)

# Avoid tokenizer parallelism + DataLoader fork warning/deadlocks
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def get_audio_embedding(model, audio_features: torch.Tensor) -> torch.Tensor:

    last_hidden_state = model.audio_tower(audio_features)[0]

    embedding = last_hidden_state.mean(dim=1)
    return embedding

def load_mask(mask_path: str) -> np.ndarray:
    data = np.load(mask_path)
    if not isinstance(data, np.lib.npyio.NpzFile):
        raise ValueError(f"Expected a .npz file at {mask_path}")
    if "mask" not in data.files:
        try:
            data.close()
        except Exception:
            pass
        raise KeyError(f"'mask' key not found in {mask_path}. Available keys: {data.files}")
    arr = np.asarray(data["mask"], dtype=np.float32).reshape(-1)
    try:
        data.close()
    except Exception:
        pass
    return arr

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, default="../data/models/Qwen2-Audio-7B-Instruct")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--mask_path", type=str, default="./logs/top48_weighted.npz")
    parser.add_argument("--train_file", type=str, default="./train.jsonl")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--tile_frames", type=int, default=3000)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--save_dir", type=str, default="./logs/checkpoints")
    parser.add_argument("--resume_tile", type=str, default=None)
    parser.add_argument("--prompt", type=str, default="Provide a detailed answer that exceeds 500 words.")
    parser.add_argument("--target", type=str, default="Sure! Here is a step-by-step guide:")
    parser.add_argument("--use_amp", action="store_true", default=True)
    parser.add_argument("--amp_dtype", type=str, choices=["fp16", "bf16"], default="bf16")
    parser.add_argument("--use_swanlab", action="store_true", default=True)
    parser.add_argument("--swan_project", type=str, default="")
    parser.add_argument("--swan_run_name", type=str, default="")
    parser.add_argument("--update_every", type=int, default=1)
    parser.add_argument("--early_stop_patience", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--lr_patience", type=int, default=5)
    parser.add_argument("--semantic_loss_weight", type=float, default=5.0)
    parser.add_argument("--clip_grad_norm", type=float, default=1.0)

    return parser.parse_args()


# =============== Tile utilities ===============
def init_tile(n_mels: int, tile_frames: int, device: torch.device, std: float = 0.01) -> torch.Tensor:
    tile = torch.randn(1, n_mels, int(tile_frames), device=device) * float(std)
    tile.requires_grad_(True)
    return tile

def apply_mask_and_clamp(x: torch.Tensor, mask_1xFx1: torch.Tensor, tau: float) -> torch.Tensor:
    x = torch.clamp(x, min=-float(tau), max=float(tau))
    return x * mask_1xFx1

# =============== Helpers ===============
def save_tile(path: str, tile: torch.Tensor, meta: dict | None = None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {"tile": tile.detach().cpu()}
    if meta:
        payload.update({"meta": meta})
    torch.save(payload, path)

def load_tile(path: str, device: torch.device, expect_frames: int | None = None) -> torch.Tensor:
    obj = torch.load(path, map_location=device)
    if isinstance(obj, torch.Tensor):
        t = obj
    elif isinstance(obj, dict) and "tile" in obj:
        t = obj["tile"]
    else:
        raise ValueError(f"Unsupported tile checkpoint format: {path}")
    t = t.to(device)
    if t.dim() != 3 or t.size(0) != 1:
        raise ValueError("Loaded tile has wrong shape; expect [1, F, L]")
    if expect_frames is not None and t.size(-1) != expect_frames:
        t = F.interpolate(t, size=expect_frames, mode="linear", align_corners=False)
    t.requires_grad_(True)
    return t

# =============== Training ===============
def train(args: argparse.Namespace):
    # Detect DDP environment
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    use_ddp = world_size > 1 and torch.cuda.is_available()

    if use_ddp and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

    use_amp = bool(getattr(args, "use_amp", False))
    amp_dtype_name = str(getattr(args, "amp_dtype", "fp16")).lower()
    amp_dtype = torch.float16 if amp_dtype_name == "fp16" else torch.bfloat16
    torch_dtype = amp_dtype

    load_device = "cpu" if use_ddp else (args.device or None)
    processor, model, _ = load_model_and_processor(
        args.model_path,
        device=load_device,
        torch_dtype=torch_dtype,
    )

    device = torch.device(f"cuda:{local_rank}") if use_ddp else torch.device(args.device or "cuda")
    model.to(device)
    try:
        setattr(model, "device", device)
    except Exception:
        pass

    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass

    use_swan = bool(getattr(args, "use_swanlab", False)) and (swanlab is not None)
    if use_swan and ((not use_ddp) or (rank == 0)):
        run_name = args.swan_run_name or f"tile_bs{args.batch_size}_amp{amp_dtype_name}_ddp{world_size}"
        swanlab.init(project=args.swan_project, name=run_name, config=vars(args))

    mask_1d = load_mask(args.mask_path)
    mask_t = torch.from_numpy(mask_1d).to(device).to(torch.float32).view(1, -1, 1)
    n_mels = int(mask_1d.shape[0])

    import json
    if not os.path.exists(args.train_file):
        raise FileNotFoundError(f"Training file not found: {args.train_file}")
    with open(args.train_file, 'r', encoding='utf-8') as f:
        records = [json.loads(line) for line in f if line.strip()]
    if not records:
        raise RuntimeError(f"No records found in {args.train_file}")

    if args.resume_tile and os.path.exists(args.resume_tile):
        tile = load_tile(args.resume_tile, device=device, expect_frames=args.tile_frames)
    else:
        tile = init_tile(n_mels=n_mels, tile_frames=args.tile_frames, device=device, std=0.01)

    if use_ddp:
        dist.broadcast(tile.data, src=0)

    tile = torch.nn.Parameter(tile)

    tok = processor.tokenizer
    eos_token_ids: List[int] = [getattr(tok, 'eos_token_id', -1)]
    # transformers >= 4.5x exposes `eos_token_ids` as an int; on the authors'
    # pinned 4.46.3 the attribute did not exist and this branch was skipped.
    # Accept both so the collected set is the same either way.
    _extra = getattr(tok, 'eos_token_ids', None)
    if _extra:
        eos_token_ids.extend([_extra] if isinstance(_extra, int) else list(_extra))
    eos_token_ids = [int(i) for i in eos_token_ids if i is not None]

    # Length of the assistant target block in tokens, measured on the chat text
    # (audio unexpanded) so it is independent of clip length: full - prefix.
    _pt, _ft, _pl = build_chat_texts(processor, args.prompt, args.target)
    suffix_len = len(tok(_ft, add_special_tokens=False)["input_ids"]) - _pl
    print(f"[labels] supervising the trailing {suffix_len} tokens "
          f"(target={args.target!r})", flush=True)

    sr = processor.feature_extractor.sampling_rate
    dataset = AudioDataset(records, sr=sr)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False) if use_ddp else None
    dataloader = DataLoader(
        dataset, batch_size=max(1, args.batch_size), shuffle=(not use_ddp),
        sampler=sampler, num_workers=4, prefetch_factor=2, pin_memory=True,
        collate_fn=collate_fn, persistent_workers=(4 > 0), drop_last=False,
    )

    os.makedirs(args.save_dir, exist_ok=True)
    global_step = 0
    best_epoch_avg = float("inf")
    epochs_without_improvement = 0

    optimizer = torch.optim.AdamW([tile], lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=args.lr_patience, min_lr=1e-5)

    for epoch in range(args.num_epochs):
        if use_ddp and isinstance(dataloader.sampler, DistributedSampler):
            dataloader.sampler.set_epoch(epoch)

        K = max(1, int(getattr(args, "update_every", 1)))
        epoch_loss_sum, epoch_atk_loss_sum, epoch_sem_loss_sum, epoch_token_sum = 0.0, 0.0, 0.0, 0
        epoch_start = time.time()

        for bidx, (wavs, batch_records) in enumerate(dataloader):
            # Use fixed target from args for all samples
            texts, prefix_lens = [], []
            for _ in batch_records:
                _, full_text, prefix_len = build_chat_texts(processor, args.prompt, args.target)
                texts.append(full_text)
                prefix_lens.append(prefix_len)

            inputs = call_processor(processor, text=texts, wavs=wavs, sampling_rate=sr,
                                    return_tensors="pt", padding=True)
            inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
            feature_key = _detect_feature_key(inputs)
            B, n_freq, T = inputs[feature_key].shape
            
            labels = build_labels_suffix(inputs["input_ids"], suffix_len,
                                         eos_token_ids=eos_token_ids,
                                         drop_trailing_eos=True).to(device)

            clean_features_fp32 = inputs[feature_key].float()

            if T != tile.size(-1):
                interpolated_tile = F.interpolate(tile, size=T, mode='linear', align_corners=False)
            else:
                interpolated_tile = tile
            expanded = apply_mask_and_clamp(interpolated_tile, mask_t, tau=args.tau)
            perturb = expanded.expand(B, -1, -1)
            mixed_input_fp32 = clean_features_fp32 + perturb.float()

            # Semantic Loss Calculation
            with torch.no_grad():
                clean_embedding = get_audio_embedding(model, clean_features_fp32.to(torch_dtype))
            perturbed_embedding = get_audio_embedding(model, mixed_input_fp32.to(torch_dtype))
            semantic_loss = F.mse_loss(clean_embedding.float(), perturbed_embedding.float())

            # Attack Loss Calculation
            with autocast('cuda', dtype=amp_dtype, enabled=use_amp):
                run_inputs = dict(inputs)
                run_inputs[feature_key] = mixed_input_fp32.to(torch_dtype)
                run_inputs["labels"] = labels
                run_inputs["use_cache"] = False
                out = model(**run_inputs)
                attack_loss = out.loss

            total_loss = attack_loss + args.semantic_loss_weight * semantic_loss

            loss_to_backprop = total_loss / float(K)
            loss_to_backprop.backward()

            # Mask the gradient immediately after backward pass and before optimizer step
            if tile.grad is not None:
                tile.grad.mul_(mask_t)

            sup_tokens = int((labels != -100).sum().item())
            epoch_loss_sum += float(total_loss.item()) * max(1, sup_tokens)
            epoch_atk_loss_sum += float(attack_loss.item()) * max(1, sup_tokens)
            # Handle both tensor and scalar cases for semantic_loss
            weighted_sem_loss = args.semantic_loss_weight * semantic_loss
            epoch_sem_loss_sum += float(weighted_sem_loss.item() if isinstance(weighted_sem_loss, torch.Tensor) else weighted_sem_loss) * max(1, sup_tokens)
            epoch_token_sum += max(1, sup_tokens)
            global_step += 1

            if ((bidx + 1) % K) == 0 or ((bidx + 1) == len(dataloader)):
                if use_ddp and (tile.grad is not None):
                    dist.all_reduce(tile.grad, op=dist.ReduceOp.SUM)
                    tile.grad.div_(world_size)
                
                if args.clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_([tile], max_norm=args.clip_grad_norm)
                
                optimizer.step()
                optimizer.zero_grad()
                with torch.no_grad():
                    tile.clamp_(min=-float(args.tau), max=float(args.tau))
                    tile.mul_(mask_t)

        epoch_sec = time.time() - epoch_start
        # Aggregate losses for logging
        loss_tensors = torch.tensor([epoch_loss_sum, epoch_atk_loss_sum, epoch_sem_loss_sum, epoch_token_sum], device=device, dtype=torch.float64)
        if use_ddp:
            dist.all_reduce(loss_tensors, op=dist.ReduceOp.SUM)
        
        total_loss_avg = loss_tensors[0].item() / max(1.0, loss_tensors[3].item())
        attack_loss_avg = loss_tensors[1].item() / max(1.0, loss_tensors[3].item())
        semantic_loss_avg = loss_tensors[2].item() / max(1.0, loss_tensors[3].item())

        if (not use_ddp) or (rank == 0):
            print(f"[E{epoch+1}] done. avg_loss={total_loss_avg:.4f} (atk={attack_loss_avg:.4f}, sem={semantic_loss_avg:.4f}) time={epoch_sec:.1f}s")
            if use_swan:
                swanlab.log({
                    "train/total_loss_avg": total_loss_avg,
                    "train/attack_loss_avg": attack_loss_avg,
                    "train/semantic_loss_avg": semantic_loss_avg,
                    "epoch_sec": epoch_sec,
                    "epoch": epoch + 1,
                }, step=global_step)

        if ((not use_ddp) or (rank == 0)):
            epoch_path = os.path.join(args.save_dir, f"epoch_{epoch+1:04d}.pt")
            save_tile(epoch_path, tile, meta={
                "epoch": epoch + 1,
                "global_step": global_step,
                "avg_loss": float(total_loss_avg),
                "avg_attack_loss": float(attack_loss_avg),
                "avg_semantic_loss": float(semantic_loss_avg),
            })

        scheduler.step(total_loss_avg)

        improved = (total_loss_avg < best_epoch_avg)
        if improved:
            best_epoch_avg = total_loss_avg
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if ((not use_ddp) or (rank == 0)) and improved:
            best_path = os.path.join(args.save_dir, "best.pt")
            save_tile(best_path, tile, meta={
                "epoch": epoch + 1, "best_avg_loss": best_epoch_avg
            })
            print(f"  -> best checkpoint updated: {best_path}")

        should_stop = (epochs_without_improvement >= args.early_stop_patience)
        if use_ddp:
            stop_t = torch.tensor([1 if should_stop else 0], device=device, dtype=torch.int32)
            dist.all_reduce(stop_t, op=dist.ReduceOp.SUM)
            should_stop = (stop_t.item() > 0)

        if should_stop:
            if (not use_ddp) or (rank == 0):
                print(f"[EARLY STOP] No improvement for {args.early_stop_patience} epochs.")
            break

    if (not use_ddp) or (rank == 0):
        print(f"[DONE] Training finished. Best checkpoint at: {os.path.join(args.save_dir, 'best.pt')}")

    if use_ddp and dist.is_initialized():
        dist.destroy_process_group()


def main():
    torch.manual_seed(42)
    args = parse_args()
    train(args)

if __name__ == "__main__":
    main()
