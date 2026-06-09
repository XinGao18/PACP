from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.prompts.prompt_bank import LearnablePromptBank
from models.prototypes.class_logits import fuse_class_logits, prompt_similarity, prototype_similarity


class PromptAlignedClassPrototype(nn.Module):
    def __init__(
        self,
        class_names: Sequence[str],
        feature_dim: int = 512,
        clip_model: str = "ViT-B/16",
        clip_download_root: str = "models/clip",
        context_length: int = 8,
        template: str = "{} a surveillance video of {}",
        alpha: float = 1.0,
        tau_p: float = 10.0,
        tau_e: float = 10.0,
    ) -> None:
        super().__init__()
        self.class_names = tuple(class_names)
        self.alpha = float(alpha)
        self.tau_p = float(tau_p)
        self.tau_e = float(tau_e)
        self.prompt_bank = LearnablePromptBank(
            class_names=self.class_names,
            clip_model=clip_model,
            clip_download_root=clip_download_root,
            context_length=context_length,
            template=template,
        )
        self.visual_prototypes = nn.Parameter(torch.empty(len(self.class_names), feature_dim))
        nn.init.normal_(self.visual_prototypes, std=0.02)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        prompt_embeddings = self.prompt_bank().to(device=features.device, dtype=features.dtype)
        prototypes = F.normalize(self.visual_prototypes.to(features.dtype), dim=-1)
        prototype_logits = prototype_similarity(features, prototypes, self.tau_p)
        prompt_logits = prompt_similarity(features, prompt_embeddings, self.tau_e)
        class_logits = fuse_class_logits(prototype_logits, prompt_logits, self.alpha)
        class_probabilities = class_logits.softmax(dim=-1)
        return {
            "prompt_embeddings": prompt_embeddings,
            "visual_prototypes": prototypes,
            "prototype_logits": prototype_logits,
            "prompt_logits": prompt_logits,
            "class_logits": class_logits,
            "class_probabilities": class_probabilities,
        }
