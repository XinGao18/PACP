from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalEnhancementHead(nn.Module):
    def __init__(
        self,
        dim: int = 512,
        hidden_dim: int | None = None,
        num_layers: int = 1,
        kernel_size: int = 3,
        num_heads: int = 4,
        dropout: float = 0.0,
        normalize: bool = True,
        head_type: str = "conv",
    ) -> None:
        super().__init__()
        self.head_type = normalize_head_type(head_type)
        self.normalize = normalize
        self.norm = nn.LayerNorm(dim)
        hidden_dim = hidden_dim or dim
        if self.head_type == "conv":
            padding = kernel_size // 2
            layers = []
            in_dim = dim
            for _ in range(num_layers):
                layers.extend(
                    [
                        nn.Conv1d(in_dim, hidden_dim, kernel_size=kernel_size, padding=padding),
                        nn.GELU(),
                        nn.Dropout(dropout),
                    ]
                )
                in_dim = hidden_dim
            layers.append(nn.Conv1d(hidden_dim, dim, kernel_size=1))
            self.net = nn.Sequential(*layers)
        else:
            layer = nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
                norm_first=True,
            )
            self.net = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim not in {2, 3}:
            raise ValueError("Expected temporal input shape [T, D] or [B, T, D]")
        has_batch = x.ndim == 3
        sequence = x if has_batch else x.unsqueeze(0)
        residual = sequence
        if self.head_type == "conv":
            enhanced = self.net(sequence.transpose(1, 2)).transpose(1, 2)
        else:
            enhanced = self.net(sequence)
        enhanced = self.norm(enhanced + residual)
        if self.normalize:
            enhanced = F.normalize(enhanced, dim=-1)
        return enhanced if has_batch else enhanced.squeeze(0)


def normalize_head_type(head_type: str) -> str:
    value = head_type.lower().replace("-", "_")
    if value in {"conv", "temporal_conv", "tcn"}:
        return "conv"
    if value in {"transformer", "temporal_transformer"}:
        return "transformer"
    raise ValueError(f"Unsupported temporal head type: {head_type}")
