import argparse
import json
import os
import time
from typing import List, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch.amp import autocast

import whisper
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

try:
    import swanlab
except Exception:
    swanlab = None

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


class AudioDataset(Dataset):
    def __init__(self, records: List[Dict], base_dir: str):
        self.records = records
        self.base_dir = base_dir

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        ap = rec.get("audio_path") or rec.get("audio")
        if not isinstance(ap, str) or not ap:
            raise ValueError("record missing audio_path/audio")
        if not os.path.isabs(ap):
            ap_abs = os.path.join(self.base_dir, ap)
        else:
            ap_abs = ap

        wav = whisper.load_audio(ap_abs)  # float32, 16k
        return wav, rec


def collate_fn(batch):
    wavs, recs = zip(*batch)
    return list(wavs), list(recs)


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


def build_chat_texts(processor: AutoProcessor, prompt: str, target: str) -> Tuple[str, str, int]:
    prompt_template = "Instruction: {query} \nFollow the text instruction based on the following audio: {speech}"
    speech = getattr(processor, "speech_token", "<SpeechHere>")

    conv_user = [{"role": "user", "content": prompt_template.format(query=prompt, speech=speech)}]
    conv_full = conv_user + [{"role": "assistant", "content": target}]

    prompt_text = processor.tokenizer.apply_chat_template(conv_user, tokenize=False, add_generation_prompt=True)
    full_text = processor.tokenizer.apply_chat_template(conv_full, tokenize=False, add_generation_prompt=False)

    prefix_ids = processor.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = processor.tokenizer(full_text, add_special_tokens=False)["input_ids"]
    prefix_len = int(len(prefix_ids))
    if prefix_len > len(full_ids):
        prefix_len = int(len(full_ids))
    return prompt_text, full_text, prefix_len


def build_labels(input_ids: torch.Tensor, prefix_len: int) -> torch.Tensor:
    labels = input_ids.clone()
    labels[:, : int(prefix_len)] = -100
    return labels


def get_chunk_audio_embedding(model, input_features: torch.Tensor, feature_attention_mask: torch.Tensor) -> torch.Tensor:
    """Return per-chunk embedding from MERaLioN speech_encoder: mean over time."""
    speech_encoder = model.speech_encoder
    out = speech_encoder(input_features, attention_mask=feature_attention_mask)
    h = out.last_hidden_state  # [N, T', C]
    emb = h.mean(dim=1)  # [N, C]
    return emb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, default="../data/models/MERaLioN")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--mask_path", type=str, default="./Meralion/logs/64_noise/top64_weighted.npz")
    parser.add_argument("--train_file", type=str, default="./train.jsonl")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--tile_frames", type=int, default=3000)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--save_dir", type=str, default="./Meralion/logs/64_noise/checkpoints")
    parser.add_argument("--resume_tile", type=str, default=None)
    parser.add_argument("--prompt", type=str, default="Provide a detailed answer that exceeds 500 words.")
    parser.add_argument("--target", type=str, default="Sure! Here is a step-by-step guide:")
    parser.add_argument("--use_amp", action="store_true", default=True)
    parser.add_argument("--amp_dtype", type=str, choices=["fp16", "bf16"], default="bf16")
    parser.add_argument("--use_swanlab", action="store_true", default=True)
    parser.add_argument("--swan_project", type=str, default="MM")
    parser.add_argument("--swan_run_name", type=str, default="meralion_tile")
    parser.add_argument("--update_every", type=int, default=1)
    parser.add_argument("--early_stop_patience", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--lr_patience", type=int, default=5)
    parser.add_argument("--semantic_loss_weight", type=float, default=5.0)
    parser.add_argument("--clip_grad_norm", type=float, default=1.0)

    return parser.parse_args()


def train(args: argparse.Namespace):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    use_ddp = world_size > 1 and torch.cuda.is_available()

    if use_ddp and not dist.is_initialized():
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")

    device = torch.device(f"cuda:{local_rank}" if use_ddp else torch.device(args.device))

    use_amp = bool(getattr(args, "use_amp", False))
    amp_dtype_name = str(getattr(args, "amp_dtype", "bf16")).lower()
    amp_dtype = torch.float16 if amp_dtype_name == "fp16" else torch.bfloat16
    torch_dtype = amp_dtype

    processor = AutoProcessor.from_pretrained(
        args.model_path,
        local_files_only=True,
        trust_remote_code=True,
    )

    device_map = f"cuda:{local_rank}" if use_ddp else "auto"
    max_memory = {local_rank: "45GiB"} if torch.cuda.is_available() else None

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        args.model_path,
        local_files_only=True,
        trust_remote_code=True,
        attn_implementation="eager",
        torch_dtype=torch_dtype,
        device_map=device_map,
        max_memory=max_memory,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    try:
        model.gradient_checkpointing_enable()
    except Exception:
        pass

    use_swan = bool(getattr(args, "use_swanlab", False)) and (swanlab is not None)
    if use_swan and ((not use_ddp) or (rank == 0)):
        run_name = args.swan_run_name or f"meralion_tile_bs{args.batch_size}_amp{amp_dtype_name}_ddp{world_size}"
        swanlab.init(project=args.swan_project, name=run_name, config=vars(args))

    # Load mask (length should be 128)
    mask_1d = load_mask(args.mask_path)
    mask_t = torch.from_numpy(mask_1d).to(device).to(torch.float32).view(1, -1, 1)
    n_mels = int(mask_1d.shape[0])

    # Load dataset records
    if not os.path.exists(args.train_file):
        raise FileNotFoundError(f"Training file not found: {args.train_file}")
    with open(args.train_file, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    if not records:
        raise RuntimeError(f"No records found in {args.train_file}")

    base_dir = os.path.dirname(os.path.expanduser(args.train_file))
    dataset = AudioDataset(records, base_dir=base_dir)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False) if use_ddp else None
    dataloader = DataLoader(
        dataset,
        batch_size=max(1, args.batch_size),
        shuffle=(not use_ddp),
        sampler=sampler,
        num_workers=4,
        prefetch_factor=2,
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=(4 > 0),
        drop_last=False,
    )

    # Tile init/resume
    os.makedirs(args.save_dir, exist_ok=True)
    if args.resume_tile and os.path.exists(args.resume_tile):
        tile = load_tile(args.resume_tile, device=device, expect_frames=args.tile_frames)
    else:
        tile = init_tile(n_mels=n_mels, tile_frames=args.tile_frames, device=device, std=0.01)

    if use_ddp:
        dist.broadcast(tile.data, src=0)

    tile = torch.nn.Parameter(tile)

    optimizer = torch.optim.AdamW([tile], lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", factor=0.5, patience=args.lr_patience, min_lr=1e-5
    )

    global_step = 0
    best_epoch_avg = float("inf")
    epochs_without_improvement = 0

    for epoch in range(args.num_epochs):
        if use_ddp and isinstance(dataloader.sampler, DistributedSampler):
            dataloader.sampler.set_epoch(epoch)

        K = max(1, int(getattr(args, "update_every", 1)))
        epoch_loss_sum, epoch_atk_loss_sum, epoch_sem_loss_sum, epoch_token_sum = 0.0, 0.0, 0.0, 0
        epoch_start = time.time()

        for bidx, (wavs, batch_records) in enumerate(dataloader):
            # Build per-sample chat text + prefix lens
            texts, prefix_lens = [], []
            for _ in batch_records:
                prompt_text, full_text, prefix_len = build_chat_texts(processor, args.prompt, args.target)
                texts.append(full_text)
                prefix_lens.append(prefix_len)

            # Processor: note MERaLioN will internally chunk audios and expand speech tokens
            inputs = processor(
                text=texts,
                audios=wavs,
                padding=True,
                sampling_rate=16000,
            )

            # Move tensors to device
            for k, v in list(inputs.items()):
                if isinstance(v, torch.Tensor):
                    if k == "input_features":
                        inputs[k] = v.to(device=device, dtype=torch_dtype)
                    else:
                        inputs[k] = v.to(device)

            # Build labels on token space
            input_ids = inputs["input_ids"]
            labels = torch.full_like(input_ids, -100)

            target_ids = processor.tokenizer(args.target, add_special_tokens=False)["input_ids"]
            if not isinstance(target_ids, list) or len(target_ids) == 0:
                raise ValueError("Tokenized target is empty; check --target")

            target_ids_t = torch.tensor(target_ids, device=input_ids.device, dtype=input_ids.dtype)

            def _find_subsequence(haystack: torch.Tensor, needle: torch.Tensor) -> int:
                h_len = int(haystack.numel())
                n_len = int(needle.numel())
                if n_len == 0 or n_len > h_len:
                    return -1
                for s in range(h_len - n_len + 1):
                    if torch.equal(haystack[s : s + n_len], needle):
                        return s
                return -1

            start_positions = []
            for i in range(input_ids.size(0)):
                seq = input_ids[i]
                pos = _find_subsequence(seq, target_ids_t)
                start_positions.append(pos)
                if pos >= 0:
                    labels[i, pos : pos + int(target_ids_t.numel())] = input_ids[i, pos : pos + int(target_ids_t.numel())]

            labels = labels.to(device)

            # Prepare features and masks
            feats = inputs["input_features"]  # [N_chunks_total, 128, 3000]
            feat_mask = inputs.get("feature_attention_mask", None)
            if feat_mask is None:
                raise KeyError("MERaLioN processor did not return feature_attention_mask")

            # Inject tile (broadcast to all chunks)
            tile_use = tile
            if feats.size(-1) != tile.size(-1):
                tile_use = F.interpolate(tile, size=int(feats.size(-1)), mode="linear", align_corners=False)
            tile_use = apply_mask_and_clamp(tile_use, mask_t, tau=args.tau)
            feats_pert = feats + tile_use.to(device=device, dtype=feats.dtype)

            semantic_loss = None
            if float(getattr(args, "semantic_loss_weight", 0.0)) > 0.0:
                with torch.no_grad():
                    clean_emb = get_chunk_audio_embedding(model, feats, feat_mask)
                pert_emb = get_chunk_audio_embedding(model, feats_pert, feat_mask)
                semantic_loss = F.mse_loss(clean_emb.float(), pert_emb.float())

            # Attack loss: CE on target text
            with autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                run_inputs = dict(inputs)
                run_inputs["input_features"] = feats_pert
                run_inputs["labels"] = labels
                run_inputs["use_cache"] = False
                out = model(**run_inputs)
                attack_loss = out.loss

            total_loss = attack_loss
            if semantic_loss is not None:
                total_loss = total_loss + float(args.semantic_loss_weight) * semantic_loss
            loss_to_backprop = total_loss / float(K)
            loss_to_backprop.backward()

            if tile.grad is not None:
                tile.grad.mul_(mask_t)

            sup_tokens = int((labels != -100).sum().item())
            epoch_loss_sum += float(total_loss.item()) * max(1, sup_tokens)
            epoch_atk_loss_sum += float(attack_loss.item()) * max(1, sup_tokens)
            if semantic_loss is not None:
                epoch_sem_loss_sum += float((args.semantic_loss_weight * semantic_loss).item()) * max(1, sup_tokens)
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

        loss_tensors = torch.tensor(
            [epoch_loss_sum, epoch_atk_loss_sum, epoch_sem_loss_sum, epoch_token_sum],
            device=device,
            dtype=torch.float64,
        )
        if use_ddp:
            dist.all_reduce(loss_tensors, op=dist.ReduceOp.SUM)

        total_loss_avg = loss_tensors[0].item() / max(1.0, loss_tensors[3].item())
        attack_loss_avg = loss_tensors[1].item() / max(1.0, loss_tensors[3].item())
        semantic_loss_avg = loss_tensors[2].item() / max(1.0, loss_tensors[3].item())

        if (not use_ddp) or (rank == 0):
            print(
                f"[E{epoch+1}] done. avg_loss={total_loss_avg:.4f} (atk={attack_loss_avg:.4f}, sem={semantic_loss_avg:.4f}) time={epoch_sec:.1f}s"
            )
            if use_swan:
                swanlab.log(
                    {
                        "train/total_loss_avg": total_loss_avg,
                        "train/attack_loss_avg": attack_loss_avg,
                        "train/semantic_loss_avg": semantic_loss_avg,
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
                    "avg_semantic_loss": float(semantic_loss_avg),
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

        should_stop = epochs_without_improvement >= args.early_stop_patience
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
