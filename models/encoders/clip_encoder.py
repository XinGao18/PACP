from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.clip import clip


class CLIPImageEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = "ViT-B/16",
        device: Optional[torch.device | str] = None,
        use_amp: bool = True,
        download_root: str | Path = "models/clip",
    ) -> None:
        super().__init__()
        self.model_name = normalize_clip_name(model_name)
        self.use_amp = use_amp
        self.download_root = str(download_root)
        self.device_hint = torch.device(device) if device is not None else None
        self.model: Optional[nn.Module] = None

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        original_shape = frames.shape
        flat_frames = flatten_video_batch(frames)
        model = self._load_model(flat_frames.device)
        with torch.no_grad():
            with torch.autocast(device_type=flat_frames.device.type, enabled=self.use_amp and flat_frames.device.type == "cuda"):
                features = model.encode_image(flat_frames)
        features = F.normalize(features.float(), dim=-1)
        return restore_video_batch(features, original_shape)

    def _load_model(self, device: torch.device) -> nn.Module:
        if self.model is None:
            self.model, _ = clip.load(self.model_name, device=self.device_hint or device, jit=False, download_root=self.download_root)
            self.model.eval()
            for parameter in self.model.parameters():
                parameter.requires_grad_(False)
        if next(self.model.parameters()).device != device:
            self.model.to(device)
        self.model.eval()
        return self.model


def normalize_clip_name(model_name: str) -> str:
    aliases = {"ViT-B-16": "ViT-B/16", "ViT-B/16": "ViT-B/16", "ViT-L-14": "ViT-L/14", "ViT-L/14": "ViT-L/14"}
    return aliases.get(model_name, model_name)


def flatten_video_batch(frames: torch.Tensor) -> torch.Tensor:
    if frames.ndim == 4:
        return frames
    if frames.ndim == 5:
        batch, time, channels, height, width = frames.shape
        return frames.reshape(batch * time, channels, height, width)
    raise ValueError(f"Expected frames shape [N,C,H,W] or [B,T,C,H,W], got {tuple(frames.shape)}")


def restore_video_batch(features: torch.Tensor, original_shape: torch.Size) -> torch.Tensor:
    if len(original_shape) == 5:
        batch, time = original_shape[:2]
        return features.reshape(batch, time, -1)
    return features
