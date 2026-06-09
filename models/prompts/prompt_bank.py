from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.clip import clip
from models.encoders.clip_encoder import normalize_clip_name


class LearnablePromptBank(nn.Module):
    def __init__(
        self,
        class_names: Sequence[str],
        clip_model: str = "ViT-B/16",
        clip_download_root: str | Path = "models/clip",
        context_length: int = 8,
        template: str = "{} a surveillance video of {}",
        normalize: bool = True,
    ) -> None:
        super().__init__()
        self.class_names = tuple(class_names)
        self.context_length = int(context_length)
        self.template = template
        self.normalize = normalize
        self.clip, _ = clip.load(normalize_clip_name(clip_model), device="cpu", jit=False, download_root=str(clip_download_root))
        self.clip.eval()
        self.clip.visual = nn.Identity()
        for parameter in self.clip.parameters():
            parameter.requires_grad_(False)
        prompts = [self.template.format(" ".join(["X"] * self.context_length), class_name) for class_name in self.class_names]
        self.register_buffer("tokenized_prompts", clip.tokenize(prompts, truncate=True), persistent=False)
        width = int(self.clip.token_embedding.embedding_dim)
        self.context_tokens = nn.Parameter(torch.empty(self.context_length, width))
        nn.init.normal_(self.context_tokens, std=0.02)

    def forward(self) -> torch.Tensor:
        device = self.context_tokens.device
        self.clip.to(device)
        tokenized = self.tokenized_prompts.to(device)
        embeddings = self.clip.token_embedding(tokenized)
        text_dtype = embeddings.dtype
        context = self.context_tokens.to(dtype=text_dtype).unsqueeze(0)
        embeddings[:, 1 : 1 + self.context_length] = context
        embeddings = embeddings + self.clip.positional_embedding.to(dtype=text_dtype)
        embeddings = embeddings.permute(1, 0, 2)
        embeddings = self.clip.transformer(embeddings)
        embeddings = embeddings.permute(1, 0, 2)
        embeddings = self.clip.ln_final(embeddings).to(dtype=text_dtype)
        pooled = embeddings[torch.arange(embeddings.shape[0], device=device), tokenized.argmax(dim=-1)] @ self.clip.text_projection
        pooled = pooled.float()
        return F.normalize(pooled, dim=-1) if self.normalize else pooled
