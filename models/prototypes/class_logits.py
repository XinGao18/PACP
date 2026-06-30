from __future__ import annotations

import torch
import torch.nn.functional as F


def prototype_similarity(features: torch.Tensor, prototypes: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    features = F.normalize(features, dim=-1)
    prototypes = F.normalize(prototypes, dim=-1)
    if prototypes.ndim == 2:
        return float(temperature) * torch.einsum("btd,cd->btc", features, prototypes)
    if prototypes.ndim == 3:
        return float(temperature) * torch.einsum("btd,cpd->btcp", features, prototypes)
    raise ValueError(f"Expected prototypes with shape [C,D] or [C,P,D], got {tuple(prototypes.shape)}")


def aggregate_prototype_logits(
    logits: torch.Tensor,
    prototype_mask: torch.Tensor | None = None,
    mode: str = "softmax",
    temperature: float = 1.0,
) -> torch.Tensor:
    if logits.ndim == 3:
        return logits
    if logits.ndim != 4:
        raise ValueError(f"Expected logits with shape [B,T,C] or [B,T,C,P], got {tuple(logits.shape)}")
    if prototype_mask is not None:
        mask = prototype_mask.to(device=logits.device, dtype=torch.bool).unsqueeze(0).unsqueeze(0)
        logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
    if mode == "max":
        return logits.max(dim=-1).values
    if mode == "mean":
        if prototype_mask is None:
            return logits.mean(dim=-1)
        weights = prototype_mask.to(device=logits.device, dtype=logits.dtype).unsqueeze(0).unsqueeze(0)
        return (logits * weights).sum(dim=-1) / weights.sum(dim=-1).clamp_min(1.0)
    if mode == "softmax":
        weights = (logits / float(temperature)).softmax(dim=-1)
        return (weights * logits).sum(dim=-1)
    if mode in {"logsumexp", "lse"}:
        return torch.logsumexp(logits, dim=-1)
    raise ValueError(f"Unsupported multi_proto_aggregation: {mode}")


def prompt_similarity(features: torch.Tensor, prompt_embeddings: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    features = F.normalize(features, dim=-1)
    prompt_embeddings = F.normalize(prompt_embeddings, dim=-1)
    return float(temperature) * torch.einsum("btd,cd->btc", features, prompt_embeddings)


def fuse_class_logits(prototype_logits: torch.Tensor, prompt_logits: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    if prototype_logits.shape != prompt_logits.shape:
        raise ValueError(f"Logit shapes must match, got {prototype_logits.shape} and {prompt_logits.shape}")
    return prototype_logits + float(alpha) * prompt_logits
