from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


class NpyFeatureDataset(Dataset):
    def __init__(self, feature_dir: str | Path, feature_key: str = "appearance") -> None:
        self.feature_dir = Path(feature_dir)
        self.feature_key = feature_key
        if not self.feature_dir.exists():
            raise FileNotFoundError(f"feature_dir not found: {self.feature_dir}")
        self.files = self._load_files()
        if not self.files:
            raise FileNotFoundError(f"No .npy feature files found in {self.feature_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        path = self.files[index]
        payload = np.load(path, allow_pickle=True).item()
        if self.feature_key not in payload:
            raise KeyError(f"Feature key {self.feature_key!r} not found in {path}")
        features = torch.as_tensor(payload[self.feature_key], dtype=torch.float32)
        if features.ndim != 2:
            raise ValueError(f"Expected feature shape [T,D] in {path}, got {tuple(features.shape)}")
        frame_indices = torch.as_tensor(payload.get("frame_indices", np.arange(features.shape[0])), dtype=torch.long)
        num_frames = int(payload.get("num_frames", 0))
        if frame_indices.numel() > 0:
            num_frames = max(num_frames, int(frame_indices.max().item()) + 1)
        if num_frames <= 0:
            num_frames = features.shape[0]
        return {
            "video_id": str(payload.get("video_id", payload.get("video_name", path.stem))),
            "frames_or_features": features,
            "video_label": torch.tensor(int(payload.get("video_label", payload.get("label", 0))), dtype=torch.long),
            "class_name": str(payload.get("class_name", "")),
            "num_frames": torch.tensor(num_frames, dtype=torch.long),
            "frame_indices": frame_indices,
        }

    def _load_files(self) -> List[Path]:
        manifest = self.feature_dir / "manifest.json"
        if manifest.exists():
            with manifest.open("r", encoding="utf-8") as handle:
                entries = json.load(handle)
            return [self.feature_dir / entry["file"] for entry in entries if (self.feature_dir / entry["file"]).exists()]
        return sorted(self.feature_dir.glob("*.npy"))


def collate_feature_batch(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "video_id": [item["video_id"] for item in batch],
        "frames_or_features": torch.stack([item["frames_or_features"] for item in batch]),
        "video_label": torch.stack([item["video_label"] for item in batch]),
        "class_name": [item["class_name"] for item in batch],
        "num_frames": torch.stack([item["num_frames"] for item in batch]),
        "frame_indices": torch.stack([item["frame_indices"] for item in batch]),
    }
