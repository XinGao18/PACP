from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def resolve_topk(top_k: int | float, length: int) -> int:
    if isinstance(top_k, bool):
        raise ValueError("top_k must be int or float, not bool")
    if isinstance(top_k, int):
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")
        return min(top_k, length)
    if isinstance(top_k, float):
        if not 0 < top_k <= 1:
            raise ValueError(f"float top_k must be in (0, 1], got {top_k}")
        return max(1, min(length, math.ceil(length * top_k)))
    raise TypeError(f"top_k must be int or float, got {type(top_k)!r}")


def topk_video_scores(class_logits: torch.Tensor, top_k: int | float = 1) -> torch.Tensor:
    if class_logits.ndim != 3:
        raise ValueError(f"class_logits must have shape [B, T, C], got {tuple(class_logits.shape)}")
    k = resolve_topk(top_k, class_logits.shape[1])
    return class_logits.topk(k, dim=1).values.mean(dim=1)


class TopKMILClassificationLoss(nn.Module):
    def __init__(self, top_k: int | float = 1) -> None:
        super().__init__()
        self.top_k = top_k

    def forward(self, class_logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        labels = labels.to(device=class_logits.device).long().view(-1)
        video_scores = topk_video_scores(class_logits, self.top_k)
        return F.cross_entropy(video_scores, labels), video_scores
