from typing import List, Tuple, Dict
import numpy as np
import librosa
from torch.utils.data import Dataset, DataLoader

class AudioDataset(Dataset):
    def __init__(self, records: List[Dict], sr: int):
        self.records = records
        self.sr = int(sr)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, Dict]:
        record = self.records[idx]
        path = record["audio_path"]
        try:
            wav, _ = librosa.load(path, sr=self.sr, mono=True)
            wav = np.asarray(wav, dtype=np.float32)
        except Exception as e:
            # On error, return a silent audio array and the original record
            wav = np.zeros(self.sr, dtype=np.float32)
        return wav, record

def collate_fn(batch: List[Tuple[np.ndarray, Dict]]) -> Tuple[List[np.ndarray], List[Dict]]:
    wavs, records = zip(*batch)
    return list(wavs), list(records)

def build_dataloader(
    audio_paths: List[str],
    sr: int,
    batch_size: int,
    num_workers: int = 4,
    prefetch_factor: int = 2,
    pin_memory: bool = True,
    shuffle: bool = True,
) -> DataLoader:
    ds = AudioDataset(audio_paths, sr=sr)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        collate_fn=collate_fn,
        drop_last=False,
    )