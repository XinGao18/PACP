from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.prompts.prompt_bank import LearnablePromptBank
from models.prototypes.class_logits import aggregate_prototype_logits, fuse_class_logits, prompt_similarity, prototype_similarity


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
        num_normal_prototypes: int = 1,
        num_abnormal_prototypes: int = 1,
        normal_class_index: int = 0,
        multi_proto_aggregation: str = "softmax",
        multi_proto_temperature: float = 1.0,
        use_motion_prototypes: bool = False,
        motion_alpha: float = 0.5,
        visual_alpha: float = 1.0,
        motion_feature_dim: int | None = None,
    ) -> None:
        super().__init__()
        if not class_names:
            raise ValueError("class_names must not be empty")
        if not 0 <= int(normal_class_index) < len(class_names):
            raise ValueError(f"normal_class_index out of range: {normal_class_index}")
        if int(num_normal_prototypes) < 1 or int(num_abnormal_prototypes) < 1:
            raise ValueError("num_normal_prototypes and num_abnormal_prototypes must be positive")
        self.class_names = tuple(class_names)
        self.alpha = float(alpha)
        self.tau_p = float(tau_p)
        self.tau_e = float(tau_e)
        self.normal_class_index = int(normal_class_index)
        self.multi_proto_aggregation = multi_proto_aggregation
        self.multi_proto_temperature = float(multi_proto_temperature)
        self.use_motion_prototypes = bool(use_motion_prototypes)
        self.motion_alpha = float(motion_alpha)
        self.visual_alpha = float(visual_alpha)
        self.prompt_bank = LearnablePromptBank(
            class_names=self.class_names,
            clip_model=clip_model,
            clip_download_root=clip_download_root,
            context_length=context_length,
            template=template,
        )
        counts = [int(num_abnormal_prototypes)] * len(self.class_names)
        counts[self.normal_class_index] = int(num_normal_prototypes)
        max_count = max(counts)
        prototype_mask = torch.zeros(len(self.class_names), max_count, dtype=torch.bool)
        for class_index, count in enumerate(counts):
            prototype_mask[class_index, :count] = True
        self.register_buffer("prototype_mask", prototype_mask, persistent=False)
        self.visual_prototypes = nn.Parameter(torch.empty(len(self.class_names), max_count, feature_dim))
        nn.init.normal_(self.visual_prototypes, std=0.02)
        if self.use_motion_prototypes:
            self.motion_prototypes = nn.Parameter(torch.empty(len(self.class_names), max_count, motion_feature_dim or feature_dim))
            nn.init.normal_(self.motion_prototypes, std=0.02)
        else:
            self.register_parameter("motion_prototypes", None)

    def _load_from_state_dict(
        self,
        state_dict: dict,
        prefix: str,
        local_metadata: dict,
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        visual_key = prefix + "visual_prototypes"
        if visual_key in state_dict and self.visual_prototypes.ndim == 3:
            loaded = state_dict[visual_key]
            if loaded.ndim == 2:
                loaded = loaded.unsqueeze(1)
            if loaded.ndim == 3 and loaded.shape[1] == 1 and self.visual_prototypes.shape[1] > 1:
                loaded = loaded.expand(-1, self.visual_prototypes.shape[1], -1).clone()
            state_dict[visual_key] = loaded
        motion_key = prefix + "motion_prototypes"
        if motion_key in state_dict and self.motion_prototypes is not None:
            loaded = state_dict[motion_key]
            if loaded.ndim == 2:
                loaded = loaded.unsqueeze(1)
            if loaded.ndim == 3 and loaded.shape[1] == 1 and self.motion_prototypes.shape[1] > 1:
                loaded = loaded.expand(-1, self.motion_prototypes.shape[1], -1).clone()
            state_dict[motion_key] = loaded
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def forward(self, features: torch.Tensor, motion: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        prompt_embeddings = self.prompt_bank().to(device=features.device, dtype=features.dtype)
        prototype_mask = self.prototype_mask.to(device=features.device)
        prototypes = F.normalize(self.visual_prototypes.to(features.dtype), dim=-1)
        prototype_logits_raw = prototype_similarity(features, prototypes, self.tau_p)
        visual_prototype_logits = aggregate_prototype_logits(
            prototype_logits_raw,
            prototype_mask,
            self.multi_proto_aggregation,
            self.multi_proto_temperature,
        )
        prototype_logits = self.visual_alpha * visual_prototype_logits
        outputs = {
            "prompt_embeddings": prompt_embeddings,
            "visual_prototypes": prototypes,
            "prototype_mask": prototype_mask,
            "prototype_logits_raw": prototype_logits_raw,
            "visual_prototype_logits": visual_prototype_logits,
        }
        if self.use_motion_prototypes:
            if motion is None:
                raise ValueError("motion is required when use_motion_prototypes is true")
            motion_prototypes = F.normalize(self.motion_prototypes.to(motion.dtype), dim=-1)
            motion_logits_raw = prototype_similarity(motion, motion_prototypes, self.tau_p)
            motion_prototype_logits = aggregate_prototype_logits(
                motion_logits_raw,
                prototype_mask,
                self.multi_proto_aggregation,
                self.multi_proto_temperature,
            )
            prototype_logits = prototype_logits + self.motion_alpha * motion_prototype_logits
            outputs.update(
                {
                    "motion_prototypes": motion_prototypes,
                    "motion_prototype_logits_raw": motion_logits_raw,
                    "motion_prototype_logits": motion_prototype_logits,
                }
            )
        prompt_logits = prompt_similarity(features, prompt_embeddings, self.tau_e)
        class_logits = fuse_class_logits(prototype_logits, prompt_logits, self.alpha)
        class_probabilities = class_logits.softmax(dim=-1)
        outputs.update(
            {
                "prototype_logits": prototype_logits,
                "prompt_logits": prompt_logits,
                "class_logits": class_logits,
                "class_probabilities": class_probabilities,
            }
        )
        return outputs
