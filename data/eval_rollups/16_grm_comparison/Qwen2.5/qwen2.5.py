import argparse
import json
import math
import os
import time
from typing import Dict, List, Tuple, Optional

from torch.utils.data import Dataset

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.data import DataLoader, DistributedSampler

import sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
MM_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if MM_ROOT not in sys.path:
    sys.path.append(MM_ROOT)

from dataloader import AudioDataset, collate_fn

try:
    import swanlab
except Exception:
    swanlab = None

from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


class OmniFeatureDataset(Dataset):
    def __init__(self, records: List[Dict], feature_dir: str):
        self.records = records
        self.feature_dir = feature_dir

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        key = rec.get("key", str(idx))
        path = os.path.join(self.feature_dir, f"{key}.pt")
        obj = torch.load(path, map_location="cpu")

        feats = obj["input_features"]
        fam = obj.get("feature_attention_mask", None)
        return feats, fam, rec

def collate_omni_features(batch: List[Tuple[torch.Tensor, Optional[torch.Tensor], Dict]]):
    feats_list, fam_list, recs = zip(*batch)
    feats = torch.stack(list(feats_list), dim=0)  # [B, F, T]

    fam_out = None
    if fam_list and fam_list[0] is not None:
        fam_out = torch.stack(list(fam_list), dim=0)  # [B, T]

    return feats, fam_out, list(recs)

def _detect_feature_key(inputs: Dict[str, torch.Tensor]) -> str:
    candidate_keys = ["input_features", "audio_values", "audio_features"]
    for k in candidate_keys:
        v = inputs.get(k, None)
        if isinstance(v, torch.Tensor) and torch.is_floating_point(v):
            return k
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor) and torch.is_floating_point(v):
            return k
    raise KeyError("No audio feature tensor found in processor outputs.")


def build_chat_and_prefix_len(processor: Qwen2_5OmniProcessor, prompt: str, target: str) -> Tuple[str, int]:
    conv_user_only = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio_url": "_"},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    conv_with_answer = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio_url": "_"},
                {"type": "text", "text": prompt},
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": target}]},
    ]

    prompt_text = processor.apply_chat_template(conv_user_only, add_generation_prompt=True, tokenize=False)
    full_text = processor.apply_chat_template(conv_with_answer, add_generation_prompt=False, tokenize=False)

    tok = processor.tokenizer
    prefix_ids = tok(prompt_text, add_special_tokens=False, return_tensors=None)["input_ids"]
    prefix_len = int(len(prefix_ids))
    return full_text, prefix_len


def build_labels(
    input_ids: torch.Tensor,
    prefix_len: int,
    eos_token_ids: List[int] | None = None,
    drop_trailing_eos: bool = True,
) -> torch.Tensor:
    labels = input_ids.clone()
    labels[:, :prefix_len] = -100

    if drop_trailing_eos and eos_token_ids:
        bsz = labels.size(0)
        for b in range(bsz):
            mask = labels[b] != -100
            if mask.any():
                last_idx = torch.nonzero(mask, as_tuple=False)[-1].item()
                if int(labels[b, last_idx].item()) in eos_token_ids:
                    labels[b, last_idx] = -100

    return labels


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


def init_tile(n_mels: int, tile_frames: int, device: torch.device, std: float = 0.01) -> torch.Tensor:
    tile = torch.randn(1, n_mels, int(tile_frames), device=device) * float(std)
    tile.requires_grad_(True)
    return tile


def apply_mask_and_clamp(x: torch.Tensor, mask_1xFx1: torch.Tensor, tau: float) -> torch.Tensor:
    x = torch.clamp(x, min=-float(tau), max=float(tau))
    return x * mask_1xFx1


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

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a masked mel-space universal noise tile for Qwen2.5-Omni")

    parser.add_argument("--model_path", type=str, default="../data/models/Qwen2.5-Omni-7B")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--mask_path", type=str, default="./Qwen2.5/logs/128_noise/top128_weighted.npz")
    parser.add_argument("--train_file", type=str, default="./train.jsonl")
    parser.add_argument("--feature_dir", type=str, default="./Qwen2.5/omni_features")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--tile_frames", type=int, default=3000)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--truncate_audio_features", type=int, default=3000)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--save_dir", type=str, default="./Qwen2.5/logs/128_noise/checkpoints")
    parser.add_argument("--resume_tile", type=str, default=None)
    parser.add_argument("--prompt", type=str, default="Provide a detailed answer that exceeds 500 words.")
    parser.add_argument("--target", type=str, default="Sure! Here is a step-by-step guide:")
    parser.add_argument("--use_amp", action="store_true", default=True)
    parser.add_argument("--amp_dtype", type=str, choices=["fp16", "bf16"], default="bf16")
    parser.add_argument("--use_swanlab", action="store_true", default=True)
    parser.add_argument("--swan_project", type=str, default="MM")
    parser.add_argument("--swan_run_name", type=str, default="omni_tile")
    parser.add_argument("--update_every", type=int, default=1)
    parser.add_argument("--early_stop_patience", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--lr_patience", type=int, default=5)
    parser.add_argument("--clip_grad_norm", type=float, default=1.0)
    parser.add_argument("--semantic_loss_weight", type=float, default=0.01)

    return parser.parse_args()


def get_audio_embedding(model: Qwen2_5OmniForConditionalGeneration, audio_features: torch.Tensor) -> torch.Tensor:
    if not hasattr(model, "thinker") or not hasattr(model.thinker, "get_audio_features"):
        raise AttributeError("Qwen2.5-Omni thinker does not expose get_audio_features()")

    thinker = model.thinker

    outs = []
    for i in range(int(audio_features.size(0))):
        x = audio_features[i : i + 1]  # [1, F, T]
        B, Freq, T = x.shape
        mask = torch.ones((B, T), device=x.device, dtype=torch.long)

        feats = thinker.get_audio_features(input_features=x, feature_attention_mask=mask)
        outs.append(feats.mean(dim=0, keepdim=True))

    return torch.cat(outs, dim=0)


def train(args: argparse.Namespace):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    use_ddp = world_size > 1 and torch.cuda.is_available()

    if use_ddp and not dist.is_initialized():
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", device_id=torch.device(f"cuda:{local_rank}"))

    device = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device("cpu")

    amp_dtype_name = str(getattr(args, "amp_dtype", "bf16")).lower()
    amp_dtype = torch.float16 if amp_dtype_name == "fp16" else torch.bfloat16
    use_amp = bool(getattr(args, "use_amp", False))

    if (not use_ddp) or (rank == 0):
        print(f"Loading Qwen2.5-Omni model from {args.model_path} ...")

    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=amp_dtype if use_amp else None,
    ).to(device)
    model.eval()
    model.thinker.eval()

    for p in model.parameters():
        p.requires_grad_(False)

    processor = Qwen2_5OmniProcessor.from_pretrained(args.model_path)

    if (not use_ddp) or (rank == 0):
        try:
            print(f"[rank{rank}] model_device={next(model.parameters()).device} dtype={next(model.parameters()).dtype}")
        except Exception:
            pass

    use_swan = bool(getattr(args, "use_swanlab", False)) and (swanlab is not None)
    if use_swan and ((not use_ddp) or (rank == 0)):
        run_name = args.swan_run_name or f"omni_tile_bs{args.batch_size}_amp{amp_dtype_name}_ddp{world_size}"
        swanlab.init(project=args.swan_project, name=run_name, config=vars(args))

    mask_1d = load_mask(args.mask_path)
    mask_t = torch.from_numpy(mask_1d).to(device=device, dtype=torch.float32).view(1, -1, 1)
    n_mels = int(mask_1d.shape[0])

    if not os.path.exists(args.train_file):
        raise FileNotFoundError(f"Training file not found: {args.train_file}")
    with open(args.train_file, "r", encoding="utf-8") as f:
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

    full_text_once, prefix_len_once = build_chat_and_prefix_len(processor, args.prompt, args.target)

    tok = processor.tokenizer
    eos_token_ids: List[int] = []
    if getattr(tok, "eos_token_id", None) is not None:
        eos_token_ids.append(int(tok.eos_token_id))

    extra_eos = getattr(tok, "eos_token_ids", None)
    if extra_eos is not None:
        if isinstance(extra_eos, (list, tuple, set)):
            eos_token_ids.extend([int(x) for x in extra_eos])
        else:
            eos_token_ids.append(int(extra_eos))

    # de-dup and drop Nones
    eos_token_ids = [int(x) for x in dict.fromkeys([x for x in eos_token_ids if x is not None])]

    sr = processor.feature_extractor.sampling_rate

    if getattr(args, "feature_dir", None):
        dataset = OmniFeatureDataset(records, feature_dir=str(args.feature_dir))
        collate = collate_omni_features
        # When loading features from disk, the loader is light; fewer workers often suffices.
        num_workers = 8
    else:
        dataset = AudioDataset(records, sr=sr)
        collate = collate_fn
        num_workers = 32

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True) if use_ddp else None
    dataloader = DataLoader(
        dataset,
        batch_size=max(1, int(args.batch_size)),
        shuffle=(not use_ddp),
        sampler=sampler,
        num_workers=num_workers,
        prefetch_factor=2,
        pin_memory=True,
        collate_fn=collate,
        persistent_workers=(num_workers > 0),
        drop_last=True,
    )

    os.makedirs(args.save_dir, exist_ok=True)
    optimizer = torch.optim.AdamW([tile], lr=float(args.lr), weight_decay=float(args.weight_decay))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=int(args.lr_patience), min_lr=1e-5
    )

    global_step = 0
    best_epoch_avg = float("inf")
    epochs_without_improvement = 0

    K = max(1, int(getattr(args, "update_every", 1)))

    for epoch in range(int(args.num_epochs)):
        if use_ddp and isinstance(dataloader.sampler, DistributedSampler):
            dataloader.sampler.set_epoch(epoch)

        epoch_loss_sum = 0.0
        epoch_atk_loss_sum = 0.0
        epoch_sem_loss_sum = 0.0
        epoch_token_sum = 0
        epoch_start = time.time()

        for bidx, batch in enumerate(dataloader):

            if getattr(args, "feature_dir", None):
                feats_b, fam_b, batch_records = batch
                B = int(feats_b.shape[0])
            else:
                wavs, batch_records = batch
                B = len(wavs)
            if B == 0:
                continue

            if getattr(args, "feature_dir", None):
                # Build inputs without running heavy audio feature extraction
                tok_out = processor.tokenizer(
                    [full_text_once] * B,
                    return_tensors="pt",
                    padding=True,
                    add_special_tokens=True,
                )
                inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in tok_out.items()}
                inputs["input_features"] = feats_b.to(device)
                if fam_b is not None:
                    inputs["feature_attention_mask"] = fam_b.to(device)
            else:
                inputs = processor(
                    text=[full_text_once] * B,
                    audio=wavs,
                    sampling_rate=sr,
                    return_tensors="pt",
                    padding=True,
                )
                inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}

            feature_key = _detect_feature_key(inputs)

            # Optional hard truncate in time dimension (match inference script)
            if int(getattr(args, "truncate_audio_features", 0)) and int(args.truncate_audio_features) > 0:
                target_T = int(args.truncate_audio_features)
                feats = inputs[feature_key]
                if isinstance(feats, torch.Tensor) and feats.dim() == 3 and feats.shape[-1] > target_T:
                    inputs[feature_key] = feats[..., :target_T]
                    if "feature_attention_mask" in inputs and isinstance(inputs.get("feature_attention_mask", None), torch.Tensor):
                        inputs["feature_attention_mask"] = inputs["feature_attention_mask"][:, :target_T]

            # Labels
            labels = build_labels(inputs["input_ids"], prefix_len_once, eos_token_ids=eos_token_ids, drop_trailing_eos=True)
            labels = labels.to(device)

            clean_features_fp32 = inputs[feature_key].float()

            target_T = int(getattr(args, "truncate_audio_features", 0) or 0)
            if target_T > 0:
                cur_T = int(clean_features_fp32.shape[-1])
                if cur_T > target_T:
                    clean_features_fp32 = clean_features_fp32[..., :target_T]
                elif cur_T < target_T:
                    clean_features_fp32 = F.pad(clean_features_fp32, (0, target_T - cur_T))

            # Force T to be a multiple of 200 (2*n_window) to avoid audio_tower split errors
            multiple = 200
            cur_T = int(clean_features_fp32.shape[-1])
            if cur_T % multiple != 0:
                new_T = (cur_T // multiple) * multiple
                if new_T <= 0:
                    new_T = multiple
                clean_features_fp32 = clean_features_fp32[..., :new_T]

            T = int(clean_features_fp32.shape[-1])

            # Interpolate tile to match T
            if T != tile.size(-1):
                interpolated_tile = F.interpolate(tile, size=T, mode="linear", align_corners=False)
            else:
                interpolated_tile = tile

            perturb = apply_mask_and_clamp(interpolated_tile, mask_t, tau=float(args.tau))
            mixed_input_fp32 = clean_features_fp32 + perturb

            # Optional semantic loss
            sem_w = float(getattr(args, "semantic_loss_weight", 0.0))
            semantic_loss = None
            if sem_w > 0:
                with torch.no_grad():
                    clean_emb = get_audio_embedding(model, clean_features_fp32.to(dtype=amp_dtype if use_amp else clean_features_fp32.dtype))
                pert_emb = get_audio_embedding(model, mixed_input_fp32.to(dtype=amp_dtype if use_amp else mixed_input_fp32.dtype))
                semantic_loss = F.mse_loss(clean_emb.float(), pert_emb.float())

            # Forward
            with autocast("cuda", dtype=amp_dtype, enabled=use_amp and device.type == "cuda"):
                run_inputs = dict(inputs)
                run_inputs[feature_key] = mixed_input_fp32.to(dtype=amp_dtype if use_amp else run_inputs[feature_key].dtype)
                run_inputs["labels"] = labels
                run_inputs["use_cache"] = False
                out = model.thinker(**run_inputs)
                attack_loss = out.loss

                if semantic_loss is None:
                    total_loss = attack_loss
                else:
                    total_loss = attack_loss + sem_w * semantic_loss

            (total_loss / float(K)).backward()

            # mask gradients
            if tile.grad is not None:
                tile.grad.mul_(mask_t)

            sup_tokens = int((labels != -100).sum().item())
            epoch_loss_sum += float(total_loss.item()) * max(1, sup_tokens)
            epoch_atk_loss_sum += float(attack_loss.item()) * max(1, sup_tokens)
            if semantic_loss is not None:
                epoch_sem_loss_sum += float(semantic_loss.item()) * max(1, sup_tokens)
            epoch_token_sum += max(1, sup_tokens)

            global_step += 1

            if ((bidx + 1) % K) == 0 or ((bidx + 1) == len(dataloader)):
                if use_ddp and (tile.grad is not None):
                    dist.all_reduce(tile.grad, op=dist.ReduceOp.SUM)
                    tile.grad.div_(world_size)

                if float(getattr(args, "clip_grad_norm", 0.0)) > 0:
                    torch.nn.utils.clip_grad_norm_([tile], max_norm=float(args.clip_grad_norm))

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                with torch.no_grad():
                    tile.clamp_(min=-float(args.tau), max=float(args.tau))
                    tile.mul_(mask_t)

        # Epoch logging
        epoch_sec = time.time() - epoch_start
        loss_tensors = torch.tensor(
            [epoch_loss_sum, epoch_atk_loss_sum, epoch_sem_loss_sum, float(epoch_token_sum)],
            device=device,
            dtype=torch.float64,
        )
        if use_ddp:
            dist.all_reduce(loss_tensors, op=dist.ReduceOp.SUM)

        total_loss_avg = loss_tensors[0].item() / max(1.0, loss_tensors[3].item())
        attack_loss_avg = loss_tensors[1].item() / max(1.0, loss_tensors[3].item())
        semantic_loss_avg = loss_tensors[2].item() / max(1.0, loss_tensors[3].item())
        semantic_loss_weighted_avg = float(getattr(args, "semantic_loss_weight", 0.0)) * semantic_loss_avg

        if (not use_ddp) or (rank == 0):
            print(
                f"[E{epoch+1}] avg_loss={total_loss_avg:.4f} "
                f"(atk={attack_loss_avg:.4f}, sem_w={semantic_loss_weighted_avg:.4f}) time={epoch_sec:.1f}s"
            )
            if use_swan:
                swanlab.log(
                    {
                        "train/total_loss_avg": total_loss_avg,
                        "train/attack_loss_avg": attack_loss_avg,
                        "train/semantic_loss_avg": semantic_loss_weighted_avg,
                        "epoch_sec": epoch_sec,
                        "epoch": epoch + 1,
                    },
                    step=global_step,
                )

        if (not use_ddp) or (rank == 0):
            epoch_path = os.path.join(args.save_dir, f"epoch_{epoch+1:04d}.pt")
            save_tile(
                epoch_path,
                tile,
                meta={
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "avg_loss": float(total_loss_avg),
                    "avg_attack_loss": float(attack_loss_avg),
                    "avg_semantic_loss_weighted": float(semantic_loss_weighted_avg),
                },
            )

        scheduler.step(total_loss_avg)

        improved = total_loss_avg < best_epoch_avg
        if improved:
            best_epoch_avg = total_loss_avg
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if ((not use_ddp) or (rank == 0)) and improved:
            best_path = os.path.join(args.save_dir, "best.pt")
            save_tile(best_path, tile, meta={"epoch": epoch + 1, "best_avg_loss": best_epoch_avg})
            print(f"  -> best checkpoint updated: {best_path}")

        should_stop = epochs_without_improvement >= int(args.early_stop_patience)
        if use_ddp:
            stop_t = torch.tensor([1 if should_stop else 0], device=device, dtype=torch.int32)
            dist.all_reduce(stop_t, op=dist.ReduceOp.SUM)
            should_stop = stop_t.item() > 0

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
