from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.encoders.clip_encoder import CLIPImageEncoder
from models.temporal.temporal_head import TemporalEnhancementHead


def feature_temporal_difference(features: torch.Tensor, first_mode: str = "zero", normalize: bool = True) -> torch.Tensor:
    if features.ndim not in {2, 3}:
        raise ValueError(f"Expected feature shape [T,D] or [B,T,D], got {tuple(features.shape)}")
    has_batch = features.ndim == 3
    sequence = features if has_batch else features.unsqueeze(0)
    motion = torch.zeros_like(sequence)
    if sequence.shape[1] > 1:
        motion[:, 1:] = sequence[:, 1:] - sequence[:, :-1]
        if first_mode == "copy":
            motion[:, 0] = motion[:, 1]
        elif first_mode != "zero":
            raise ValueError(f"Unsupported first_mode: {first_mode}")
    if normalize:
        motion = F.normalize(motion, dim=-1)
    return motion if has_batch else motion.squeeze(0)


class AppearanceMotionEncoder(nn.Module):
    def __init__(
        self,
        clip_model: str = "ViT-B/16",
        clip_download_root: str = "models/clip",
        feature_dim: int = 512,
        projection_dim: int = 512,
        first_motion_mode: str = "zero",
        normalize_motion: bool = True,
        temporal_head: Optional[nn.Module] = None,
        use_amp: bool = True,
    ) -> None:
        super().__init__()
        self.image_encoder = CLIPImageEncoder(model_name=clip_model, use_amp=use_amp, download_root=clip_download_root)
        self.first_motion_mode = first_motion_mode
        self.normalize_motion = normalize_motion
        self.projection = nn.Linear(feature_dim * 2, projection_dim)
        self.temporal_head = temporal_head or TemporalEnhancementHead(dim=projection_dim)

    def forward(self, frames_or_features: torch.Tensor, is_feature_input: Optional[bool] = None) -> Dict[str, torch.Tensor]:
        appearance = self.encode_appearance(frames_or_features, is_feature_input)
        motion = feature_temporal_difference(appearance, self.first_motion_mode, self.normalize_motion)
        fused = F.normalize(self.projection(torch.cat([appearance, motion], dim=-1)), dim=-1)
        enhanced = self.temporal_head(fused)
        return {"appearance": appearance, "motion": motion, "fused": fused, "enhanced": enhanced, "z": enhanced}

    def encode_appearance(self, frames_or_features: torch.Tensor, is_feature_input: Optional[bool]) -> torch.Tensor:
        if is_feature_input is None:
            is_feature_input = frames_or_features.ndim in {2, 3}
        if is_feature_input:
            return F.normalize(frames_or_features.float(), dim=-1)
        return self.image_encoder(frames_or_features)
