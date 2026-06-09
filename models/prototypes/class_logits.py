from __future__ import annotations

import torch
import torch.nn.functional as F


def prototype_similarity(features: torch.Tensor, prototypes: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    features = F.normalize(features, dim=-1)
    prototypes = F.normalize(prototypes, dim=-1)
    return float(temperature) * torch.einsum("btd,cd->btc", features, prototypes)


def prompt_similarity(features: torch.Tensor, prompt_embeddings: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    features = F.normalize(features, dim=-1)
    prompt_embeddings = F.normalize(prompt_embeddings, dim=-1)
    return float(temperature) * torch.einsum("btd,cd->btc", features, prompt_embeddings)


def fuse_class_logits(prototype_logits: torch.Tensor, prompt_logits: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    if prototype_logits.shape != prompt_logits.shape:
        raise ValueError(f"Logit shapes must match, got {prototype_logits.shape} and {prompt_logits.shape}")
    return prototype_logits + float(alpha) * prompt_logits
