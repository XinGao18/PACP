from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from losses.topk_mil_loss import TopKMILClassificationLoss
from losses.prototype_loss import PromptPrototypeAlignmentLoss, PrototypeSeparationLoss


@dataclass
class LossWeights:
    cls: float = 1.0
    align: float = 0.1
    sep: float = 0.1


def total_loss(l_cls: torch.Tensor, l_align: torch.Tensor, l_sep: torch.Tensor, weights: LossWeights) -> torch.Tensor:
    return weights.cls * l_cls + weights.align * l_align + weights.sep * l_sep


class TotalLoss(nn.Module):
    def __init__(self, top_k: int | float = 1, margin: float = 0.2, weights: LossWeights | None = None) -> None:
        super().__init__()
        self.classification_loss = TopKMILClassificationLoss(top_k=top_k)
        self.alignment_loss = PromptPrototypeAlignmentLoss()
        self.separation_loss = PrototypeSeparationLoss(margin=margin)
        self.weights = weights or LossWeights()

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        labels: torch.Tensor | None = None,
        batch: dict | None = None,
        **_: object,
    ) -> dict[str, torch.Tensor]:
        if labels is None:
            if batch is None:
                raise ValueError("labels or batch are required")
            labels = batch.get("video_label", batch.get("label"))
        if labels is None:
            raise ValueError("labels are required")
        l_cls, video_scores = self.classification_loss(outputs["class_logits"], labels)
        l_align = self.alignment_loss(outputs["visual_prototypes"], outputs["prompt_embeddings"])
        l_sep = self.separation_loss(outputs["visual_prototypes"])
        loss = total_loss(l_cls, l_align, l_sep, self.weights)
        return {
            "loss": loss,
            "cls": l_cls,
            "align": l_align,
            "sep": l_sep,
            "video_scores": video_scores.detach(),
        }


def build_total_loss(cfg: dict) -> TotalLoss:
    loss_cfg = cfg.get("loss", {})
    weights_cfg = loss_cfg.get("weights", {})
    prototype_cfg = cfg.get("model", {}).get("prototype", {})
    return TotalLoss(
        top_k=loss_cfg.get("top_k", 1),
        margin=float(prototype_cfg.get("margin", 0.2)),
        weights=LossWeights(
            cls=float(weights_cfg.get("cls", 1.0)),
            align=float(weights_cfg.get("align", 0.1)),
            sep=float(weights_cfg.get("sep", 0.1)),
        ),
    )
