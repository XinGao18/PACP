from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _zero_like(reference: torch.Tensor) -> torch.Tensor:
    return reference.sum() * 0.0


def _valid_prototypes(
    visual_prototypes: torch.Tensor,
    prototype_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if visual_prototypes.ndim == 2:
        class_ids = torch.arange(visual_prototypes.shape[0], device=visual_prototypes.device)
        return visual_prototypes, class_ids
    if visual_prototypes.ndim != 3:
        raise ValueError(f"Expected prototypes with shape [C,D] or [C,P,D], got {tuple(visual_prototypes.shape)}")
    if prototype_mask is None:
        prototype_mask = torch.ones(visual_prototypes.shape[:2], dtype=torch.bool, device=visual_prototypes.device)
    else:
        prototype_mask = prototype_mask.to(device=visual_prototypes.device, dtype=torch.bool)
    class_ids = torch.arange(visual_prototypes.shape[0], device=visual_prototypes.device).unsqueeze(1).expand_as(prototype_mask)
    return visual_prototypes[prototype_mask], class_ids[prototype_mask]


class PromptPrototypeAlignmentLoss(nn.Module):
    def forward(
        self,
        visual_prototypes: torch.Tensor,
        prompt_embeddings: torch.Tensor,
        prototype_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        visual_prototypes = F.normalize(visual_prototypes, dim=-1)
        prompt_embeddings = F.normalize(prompt_embeddings, dim=-1)
        if visual_prototypes.ndim == 2:
            return (1.0 - (visual_prototypes * prompt_embeddings).sum(dim=-1)).mean()
        if visual_prototypes.ndim != 3:
            raise ValueError(f"Expected visual_prototypes with shape [C,D] or [C,P,D], got {tuple(visual_prototypes.shape)}")
        if prototype_mask is None:
            prototype_mask = torch.ones(visual_prototypes.shape[:2], dtype=torch.bool, device=visual_prototypes.device)
        prototype_mask = prototype_mask.to(device=visual_prototypes.device, dtype=torch.bool)
        similarity = (visual_prototypes * prompt_embeddings.unsqueeze(1)).sum(dim=-1)
        if not prototype_mask.any():
            return _zero_like(similarity)
        return (1.0 - similarity[prototype_mask]).mean()


class PrototypeSeparationLoss(nn.Module):
    def __init__(
        self,
        margin: float = 0.2,
        normal_class_index: int = 0,
        normal_compact_weight: float = 0.1,
        normal_diversity_weight: float = 0.05,
        normal_diversity_margin: float = 0.95,
    ) -> None:
        super().__init__()
        self.margin = float(margin)
        self.normal_class_index = int(normal_class_index)
        self.normal_compact_weight = float(normal_compact_weight)
        self.normal_diversity_weight = float(normal_diversity_weight)
        self.normal_diversity_margin = float(normal_diversity_margin)

    def forward(self, visual_prototypes: torch.Tensor, prototype_mask: torch.Tensor | None = None) -> torch.Tensor:
        if visual_prototypes.ndim == 2:
            return self._inter_class_loss(visual_prototypes)
        if visual_prototypes.ndim != 3:
            raise ValueError(f"Expected visual_prototypes with shape [C,D] or [C,P,D], got {tuple(visual_prototypes.shape)}")
        prototypes, class_ids = _valid_prototypes(visual_prototypes, prototype_mask)
        inter_class = self._inter_class_loss(prototypes, class_ids)
        normal_loss = self._normal_hierarchy_loss(visual_prototypes, prototype_mask)
        return inter_class + normal_loss

    def _inter_class_loss(self, prototypes: torch.Tensor, class_ids: torch.Tensor | None = None) -> torch.Tensor:
        prototypes = F.normalize(prototypes, dim=-1)
        if prototypes.shape[0] <= 1:
            return _zero_like(prototypes)
        similarity = prototypes @ prototypes.t()
        mask = ~torch.eye(similarity.shape[0], dtype=torch.bool, device=similarity.device)
        if class_ids is not None:
            mask = mask & (class_ids.unsqueeze(0) != class_ids.unsqueeze(1))
        if not mask.any():
            return _zero_like(similarity)
        return torch.relu(similarity[mask] - self.margin).mean()

    def _normal_hierarchy_loss(self, visual_prototypes: torch.Tensor, prototype_mask: torch.Tensor | None) -> torch.Tensor:
        if not 0 <= self.normal_class_index < visual_prototypes.shape[0]:
            return _zero_like(visual_prototypes)
        normal_prototypes = visual_prototypes[self.normal_class_index]
        if prototype_mask is not None:
            normal_mask = prototype_mask.to(device=visual_prototypes.device, dtype=torch.bool)[self.normal_class_index]
            normal_prototypes = normal_prototypes[normal_mask]
        if normal_prototypes.shape[0] <= 1:
            return _zero_like(visual_prototypes)
        normal_prototypes = F.normalize(normal_prototypes, dim=-1)
        center = F.normalize(normal_prototypes.mean(dim=0, keepdim=True), dim=-1)
        compact = (1.0 - (normal_prototypes * center).sum(dim=-1)).mean()
        similarity = normal_prototypes @ normal_prototypes.t()
        pair_mask = ~torch.eye(similarity.shape[0], dtype=torch.bool, device=similarity.device)
        diversity = torch.relu(similarity[pair_mask] - self.normal_diversity_margin).mean()
        return self.normal_compact_weight * compact + self.normal_diversity_weight * diversity


class NormalVideoSuppressionLoss(nn.Module):
    def __init__(self, normal_class_index: int = 0) -> None:
        super().__init__()
        self.normal_class_index = int(normal_class_index)

    def forward(self, class_probabilities: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        labels = labels.to(device=class_probabilities.device).long().view(-1)
        normal_mask = labels == self.normal_class_index
        if not normal_mask.any():
            return _zero_like(class_probabilities)
        anomaly_scores = 1.0 - class_probabilities[..., self.normal_class_index]
        return anomaly_scores[normal_mask].mean()
