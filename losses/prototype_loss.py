from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PromptPrototypeAlignmentLoss(nn.Module):
    def forward(self, visual_prototypes: torch.Tensor, prompt_embeddings: torch.Tensor) -> torch.Tensor:
        visual_prototypes = F.normalize(visual_prototypes, dim=-1)
        prompt_embeddings = F.normalize(prompt_embeddings, dim=-1)
        return (1.0 - (visual_prototypes * prompt_embeddings).sum(dim=-1)).mean()


class PrototypeSeparationLoss(nn.Module):
    def __init__(self, margin: float = 0.2) -> None:
        super().__init__()
        self.margin = float(margin)

    def forward(self, visual_prototypes: torch.Tensor) -> torch.Tensor:
        prototypes = F.normalize(visual_prototypes, dim=-1)
        similarity = prototypes @ prototypes.t()
        mask = ~torch.eye(similarity.shape[0], dtype=torch.bool, device=similarity.device)
        if not mask.any():
            return similarity.sum() * 0.0
        return torch.relu(similarity[mask] - self.margin).mean()
