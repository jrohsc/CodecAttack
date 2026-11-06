import argparse
import json
import os
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.data import DataLoader, DistributedSampler
import whisper

import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
MM_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if MM_ROOT not in sys.path:
    sys.path.append(MM_ROOT)

from dataloader import AudioDataset, collate_fn

try:
    import swanlab
except ImportError:
    swanlab = None

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def ensure_llama_omni2_imports(llama_omni2_root: str):
    root = os.path.expanduser(llama_omni2_root)
    if not os.path.isdir(root):
        raise FileNotFoundError(f"LLaMA-Omni2 repo not found at: {root}")
    if root not in sys.path:
        sys.path.insert(0, root)


def init_tile(n_mels: int, tile_frames: int, device: torch.device, std: float = 0.01) -> torch.Tensor:
    tile = torch.randn(1, n_mels, int(tile_frames), device=device) * float(std)
    tile.requires_grad_(True)
    return tile


def apply_mask_and_clamp(x: torch.Tensor, mask_1x1xF: torch.Tensor, tau: float) -> torch.Tensor:
    x = torch.clamp(x, min=-float(tau), max=float(tau))
    return x * mask_1x1xF


def tile_repeat_crop(tile_1xFxL: torch.Tensor, target_frames: int) -> torch.Tensor:
    if tile_1xFxL.dim() != 3 or tile_1xFxL.size(0) != 1:
        raise ValueError("tile must have shape [1, F, L]")
    L = int(tile_1xFxL.size(-1))
    T = int(target_frames)
    if T <= 0:
        raise ValueError("target_frames must be positive")

    if T <= L:
        return tile_1xFxL[..., :T]

    reps = (T + L - 1) // L
    tiled = tile_1xFxL.repeat(1, 1, reps)
    return tiled[..., :T]


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


def wav_to_mel_T128(wav_np: np.ndarray) -> torch.Tensor:
    # Returns [T,128] float32 (CPU)
    wav = torch.from_numpy(np.asarray(wav_np, dtype=np.float32))
    wav = whisper.pad_or_trim(wav)
    mel = whisper.log_mel_spectrogram(wav, n_mels=128).permute(1, 0)
    return mel


def build_prompt_and_labels(
    tokenizer,
    speech_token_index: int,
    default_speech_token: str,
    prompt: str,
    target: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    user_content = default_speech_token
    if prompt:
        user_content = f"{default_speech_token} {prompt}"

    conv_user = [{"role": "user", "content": user_content}]
    conv_full = conv_user + [{"role": "assistant", "content": target}]

    prompt_ids = tokenizer.apply_chat_template(
        conv_user, add_generation_prompt=False, return_tensors="pt"
    )[0]
    full_ids = tokenizer.apply_chat_template(
        conv_full, add_generation_prompt=False, return_tensors="pt"
    )[0]

    ids_speech = tokenizer.convert_tokens_to_ids(default_speech_token)
    prompt_ids = prompt_ids.clone()
    full_ids = full_ids.clone()
    prompt_ids[prompt_ids == ids_speech] = speech_token_index
    full_ids[full_ids == ids_speech] = speech_token_index

    prefix_len = int(prompt_ids.numel())

    labels = full_ids.clone()
    labels[:prefix_len] = -100

    return full_ids.unsqueeze(0), labels.unsqueeze(0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a masked mel-space universal noise tile for LLaMA-Omni2")
    parser.add_argument("--llama_omni2_root", type=str, default="../LLaMA-Omni2")
    parser.add_argument("--model_path", type=str, default="../data/models/Llama-Omni2")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--mask_path", type=str, default="./Llama2/logs/32_noise/top32_weighted.npz")
    parser.add_argument("--train_file", type=str, default="./train.jsonl")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--tile_frames", type=int, default=3000)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--save_dir", type=str, default="./Llama2/logs/32_noise/checkpoints")
    parser.add_argument("--resume_tile", type=str, default=None)
    parser.add_argument("--prompt", type=str, default="Provide a detailed answer that exceeds 500 words.")
    parser.add_argument("--target", type=str, default="Sure! Here is a step-by-step guide:")
    parser.add_argument("--use_amp", action="store_true", default=True)
    parser.add_argument("--amp_dtype", type=str, choices=["fp16", "bf16"], default="bf16")
    parser.add_argument("--use_swanlab", action="store_true", default=True)
    parser.add_argument("--swan_project", type=str, default="MM")
    parser.add_argument("--swan_run_name", type=str, default="llama2_tile")
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
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else torch.device("cpu"))

    amp_dtype_name = str(getattr(args, "amp_dtype", "bf16")).lower()
    amp_dtype = torch.float16 if amp_dtype_name == "fp16" else torch.bfloat16
    use_amp = bool(getattr(args, "use_amp", False)) and device.type == "cuda"

    use_swan = bool(getattr(args, "use_swanlab", False)) and (swanlab is not None)
    if use_swan and ((not use_ddp) or (rank == 0)):
        run_name = args.swan_run_name or f"llama2_tile_bs{args.batch_size}_amp{amp_dtype_name}_ddp{world_size}"
        swanlab.init(project=args.swan_project, name=run_name, config=vars(args))

    ensure_llama_omni2_imports(args.llama_omni2_root)

    from transformers import AutoConfig, AutoTokenizer  
    from llama_omni2.model import Omni2SpeechQwen2ForCausalLM  # noqa: E402
    from llama_omni2.constants import SPEECH_TOKEN_INDEX, DEFAULT_SPEECH_TOKEN  # noqa: E402

    if (not use_ddp) or (rank == 0):
        print(f"Loading LLaMA-Omni2 from {args.model_path} ...")

    model_path = os.path.expanduser(args.model_path)
    config = AutoConfig.from_pretrained(model_path)
    config.tts_tokenizer = os.path.join(model_path, "tts_tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = Omni2SpeechQwen2ForCausalLM.from_pretrained(model_path, config=config, torch_dtype=amp_dtype if use_amp else None).to(device)

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    mask_1d = load_mask(args.mask_path)
    n_mels = int(mask_1d.shape[0])
    if n_mels != 128:
        raise ValueError(f"mask length must be 128 for Omni2 mel; got {n_mels}")
    mask_speech = torch.from_numpy(mask_1d).to(device=device, dtype=torch.float32).view(1, 1, -1)  # [1,1,128]
    mask_tile = torch.from_numpy(mask_1d).to(device=device, dtype=torch.float32).view(1, -1, 1)  # [1,128,1]

    if not os.path.exists(args.train_file):
        raise FileNotFoundError(f"Training file not found: {args.train_file}")
    with open(args.train_file, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    if not records:
        raise RuntimeError(f"No records found in {args.train_file}")

    # Resolve audio_path to absolute based on train_file dir
    base_dir = os.path.dirname(os.path.expanduser(args.train_file))
    for r in records:
        ap = r.get("audio_path") or r.get("audio")
        if isinstance(ap, str) and ap and (not os.path.isabs(ap)):
            r["audio_path"] = os.path.join(base_dir, ap)
        elif isinstance(ap, str):
            r["audio_path"] = ap

    if args.resume_tile and os.path.exists(args.resume_tile):
        tile = load_tile(args.resume_tile, device=device, expect_frames=args.tile_frames)
    else:
        tile = init_tile(n_mels=n_mels, tile_frames=args.tile_frames, device=device, std=0.01)

    if use_ddp:
        dist.broadcast(tile.data, src=0)

    tile = torch.nn.Parameter(tile)

    os.makedirs(args.save_dir, exist_ok=True)
    optimizer = torch.optim.AdamW([tile], lr=float(args.lr), weight_decay=float(args.weight_decay))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=int(args.lr_patience), min_lr=1e-5
    )

    global_step = 0
    best_epoch_avg = float("inf")
    epochs_without_improvement = 0
    K = max(1, int(getattr(args, "update_every", 1)))

    dataset = AudioDataset(records, sr=16000)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False) if use_ddp else None
    dataloader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=(not use_ddp),
        sampler=sampler,
        num_workers=4,
        prefetch_factor=2,
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=True,
        drop_last=False,
    )

    for epoch in range(int(args.num_epochs)):
        if use_ddp and isinstance(dataloader.sampler, DistributedSampler):
            dataloader.sampler.set_epoch(epoch)

        epoch_loss_sum = 0.0
        epoch_atk_loss_sum = 0.0
        epoch_sem_loss_sum = 0.0
        epoch_token_sum = 0
        epoch_start = time.time()

        optimizer.zero_grad(set_to_none=True)

        for bidx, (wavs, batch_records) in enumerate(dataloader):
            batch_loss = 0.0
            batch_tokens = 0 
            current_bs = len(wavs)

            for wav, rec in zip(wavs, batch_records):
                ap = rec.get("audio_path")
                gt = rec.get("gt") or rec.get("text") or rec.get("transcript") or ""

                mel = wav_to_mel_T128(wav).to(device=device, dtype=torch.float32)  # [T,128]
                T = int(mel.shape[0])

                tile_use = tile_repeat_crop(tile, target_frames=T)  # [1,128,T]
                tile_use_T128 = tile_use.transpose(1, 2).contiguous()  # [1,T,128]
                perturb = apply_mask_and_clamp(tile_use_T128, mask_speech, tau=float(args.tau))  # [1,T,128]
                perturb_T128 = perturb.squeeze(0).contiguous()  # [T,128]

                mixed = mel + perturb_T128

                target = str(getattr(args, "target", ""))

                input_ids, labels = build_prompt_and_labels(
                    tokenizer,
                    speech_token_index=SPEECH_TOKEN_INDEX,
                    default_speech_token=DEFAULT_SPEECH_TOKEN,
                    prompt=str(getattr(args, "prompt", "")),
                    target=target,
                )
                input_ids = input_ids.to(device)
                labels = labels.to(device)
                attn = torch.ones_like(input_ids, dtype=torch.long, device=device)

                speech = mixed.to(dtype=amp_dtype if use_amp else mixed.dtype).unsqueeze(0)  # [1,T,128]
                speech_lengths = torch.LongTensor([speech.shape[1]]).to(device)

                speech_clean = mel.to(dtype=amp_dtype if use_amp else mel.dtype).unsqueeze(0)  # [1,T,128]

                if float(getattr(args, "semantic_loss_weight", 0.0)) > 0:
                    with torch.no_grad():
                        clean_feats = model.encode_speech(speech_clean, speech_lengths)[0]
                        clean_emb = clean_feats.mean(dim=0)

                    pert_feats = model.encode_speech(speech, speech_lengths)[0]
                    pert_emb = pert_feats.mean(dim=0)
                    semantic_loss = F.mse_loss(clean_emb.float(), pert_emb.float())
                else:
                    semantic_loss = torch.zeros((), device=device)

                with autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                    out = model(
                        input_ids=input_ids,
                        attention_mask=attn,
                        speech=speech,
                        speech_lengths=speech_lengths,
                        labels=labels,
                        use_cache=False,
                    )
                    attack_loss_mean = out.loss

                sup_tokens = int((labels != -100).sum().item())
                attack_loss_sum = attack_loss_mean * float(max(1, sup_tokens))
                total_loss_mean = attack_loss_mean + float(getattr(args, "semantic_loss_weight", 0.0)) * semantic_loss
                total_loss_sum = total_loss_mean * float(max(1, sup_tokens))


                (total_loss_mean / float(current_bs)).backward()

                if tile.grad is not None:
                    tile.grad.mul_(mask_tile)

                sem_weight = float(getattr(args, "semantic_loss_weight", 0.0))
                sem_weighted_sum = (sem_weight * semantic_loss) * float(max(1, sup_tokens))

                batch_tokens += max(1, sup_tokens)
                batch_loss += float(total_loss_sum.detach().item())
                epoch_atk_loss_sum += float(attack_loss_sum.detach().item())
                epoch_sem_loss_sum += float(sem_weighted_sum.detach().item())

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
                    tile.mul_(mask_tile)

            epoch_loss_sum += batch_loss
            epoch_token_sum += batch_tokens

        epoch_sec = time.time() - epoch_start

        loss_tensors = torch.tensor(
            [epoch_loss_sum, epoch_atk_loss_sum, epoch_sem_loss_sum, float(epoch_token_sum)],
            device=device,
            dtype=torch.float64,
        )
        if use_ddp:
            dist.all_reduce(loss_tensors, op=dist.ReduceOp.SUM)

        token_sum = max(1.0, loss_tensors[3].item())
        avg_loss = loss_tensors[0].item() / token_sum
        avg_atk_loss = loss_tensors[1].item() / token_sum
        avg_sem_loss_weighted = loss_tensors[2].item() / token_sum

        if (not use_ddp) or (rank == 0):
            print(
                f"[E{epoch+1}] avg_loss={avg_loss:.4f} (atk={avg_atk_loss:.4f}, sem={avg_sem_loss_weighted:.4f}) time={epoch_sec:.1f}s"
            )
            if use_swan:
                swanlab.log(
                    {
                        "train/avg_loss": avg_loss,
                        "train/attack_loss_avg": avg_atk_loss,
                        "train/semantic_loss_weighted_avg": avg_sem_loss_weighted,
                        "epoch_sec": epoch_sec,
                        "epoch": epoch + 1,
                    },
                    step=global_step,
                )

        if (not use_ddp) or (rank == 0):
            epoch_path = os.path.join(args.save_dir, f"epoch_{epoch+1:04d}.pt")
            save_tile(epoch_path, tile, meta={"epoch": epoch + 1, "global_step": global_step, "avg_loss": float(avg_loss)})

        scheduler.step(avg_loss)

        improved = avg_loss < best_epoch_avg
        if improved:
            best_epoch_avg = avg_loss
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

