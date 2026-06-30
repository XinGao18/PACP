from __future__ import annotations

from typing import Any, Dict, Sequence

import torch
import torch.nn as nn

from losses.topk_mil_loss import topk_video_scores
from models.encoders import AppearanceMotionEncoder
from models.prototypes import PromptAlignedClassPrototype
from models.temporal import TemporalEnhancementHead


class PCPLModel(nn.Module):
    def __init__(
        self,
        class_names: Sequence[str],
        clip_model: str = "ViT-B-16",
        clip_download_root: str = "models/clip",
        feature_dim: int = 512,
        projection_dim: int = 512,
        first_motion_mode: str = "zero",
        normalize_motion: bool = True,
        temporal_hidden_dim: int | None = None,
        temporal_layers: int = 1,
        temporal_kernel_size: int = 3,
        temporal_dropout: float = 0.0,
        temporal_normalize: bool = True,
        temporal_type: str = "conv",
        temporal_num_heads: int = 4,
        prompt_context_length: int = 8,
        prompt_template: str = "{} a surveillance video of {}",
        prototype_alpha: float = 1.0,
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
        top_k: int | float = 4,
        use_amp: bool = True,
    ) -> None:
        super().__init__()
        self.class_names = tuple(class_names)
        self.top_k = top_k
        temporal_head = TemporalEnhancementHead(
            dim=projection_dim,
            hidden_dim=temporal_hidden_dim,
            num_layers=temporal_layers,
            kernel_size=temporal_kernel_size,
            dropout=temporal_dropout,
            normalize=temporal_normalize,
            head_type=temporal_type,
            num_heads=temporal_num_heads,
        )
        self.encoder = AppearanceMotionEncoder(
            clip_model=clip_model,
            clip_download_root=clip_download_root,
            feature_dim=feature_dim,
            projection_dim=projection_dim,
            first_motion_mode=first_motion_mode,
            normalize_motion=normalize_motion,
            temporal_head=temporal_head,
            use_amp=use_amp,
        )
        self.classifier = PromptAlignedClassPrototype(
            class_names=self.class_names,
            feature_dim=projection_dim,
            clip_model=clip_model,
            clip_download_root=clip_download_root,
            context_length=prompt_context_length,
            template=prompt_template,
            alpha=prototype_alpha,
            tau_p=tau_p,
            tau_e=tau_e,
            num_normal_prototypes=num_normal_prototypes,
            num_abnormal_prototypes=num_abnormal_prototypes,
            normal_class_index=normal_class_index,
            multi_proto_aggregation=multi_proto_aggregation,
            multi_proto_temperature=multi_proto_temperature,
            use_motion_prototypes=use_motion_prototypes,
            motion_alpha=motion_alpha,
            visual_alpha=visual_alpha,
            motion_feature_dim=feature_dim,
        )

    def forward(self, frames_or_features: torch.Tensor, is_feature_input: bool | None = None) -> Dict[str, torch.Tensor]:
        encoded = self.encoder(frames_or_features, is_feature_input=is_feature_input)
        classified = self.classifier(features=encoded["z"], motion=encoded["motion"])
        class_logits = classified["class_logits"]
        class_probabilities = classified["class_probabilities"]
        video_scores = topk_video_scores(class_logits, self.top_k)
        return {
            **encoded,
            **classified,
            "video_scores": video_scores,
            "video_probabilities": video_scores.softmax(dim=-1),
            "anomaly_scores": 1.0 - class_probabilities[..., 0],
        }


def build_pcpl_model(cfg: Dict[str, Any]) -> PCPLModel:
    model_cfg = cfg.get("model", {})
    motion_cfg = model_cfg.get("motion", {})
    temporal_cfg = model_cfg.get("temporal", {})
    prompt_cfg = model_cfg.get("prompt", {})
    prototype_cfg = model_cfg.get("prototype", {})
    class_names = tuple(cfg.get("dataset", {}).get("class_names") or prompt_cfg.get("class_names") or ["normal", "abnormal"])
    return PCPLModel(
        class_names=class_names,
        clip_model=model_cfg.get("clip_model", "ViT-B-16"),
        clip_download_root=model_cfg.get("clip_download_root", "models/clip"),
        feature_dim=int(model_cfg.get("feature_dim", 512)),
        projection_dim=int(model_cfg.get("projection_dim", model_cfg.get("feature_dim", 512))),
        first_motion_mode=motion_cfg.get("first_mode", "zero"),
        normalize_motion=bool(motion_cfg.get("normalize", True)),
        temporal_hidden_dim=int(temporal_cfg["hidden_dim"]) if temporal_cfg.get("hidden_dim") is not None else None,
        temporal_layers=int(temporal_cfg.get("num_layers", 1)),
        temporal_kernel_size=int(temporal_cfg.get("kernel_size", 3)),
        temporal_dropout=float(temporal_cfg.get("dropout", 0.0)),
        temporal_normalize=bool(temporal_cfg.get("normalize", True)),
        temporal_type=temporal_cfg.get("type", "conv"),
        temporal_num_heads=int(temporal_cfg.get("num_heads", 4)),
        prompt_context_length=int(prompt_cfg.get("context_length", 8)),
        prompt_template=prompt_cfg.get("template", "{} a surveillance video of {}"),
        prototype_alpha=float(prototype_cfg.get("alpha", 1.0)),
        tau_p=float(prototype_cfg.get("tau_p", 10.0)),
        tau_e=float(prototype_cfg.get("tau_e", 10.0)),
        num_normal_prototypes=int(prototype_cfg.get("num_normal_prototypes", 1)),
        num_abnormal_prototypes=int(prototype_cfg.get("num_abnormal_prototypes", 1)),
        normal_class_index=int(prototype_cfg.get("normal_class_index", 0)),
        multi_proto_aggregation=prototype_cfg.get("multi_proto_aggregation", "softmax"),
        multi_proto_temperature=float(prototype_cfg.get("multi_proto_temperature", 1.0)),
        use_motion_prototypes=bool(prototype_cfg.get("use_motion_prototypes", False)),
        motion_alpha=float(prototype_cfg.get("motion_alpha", 0.5)),
        visual_alpha=float(prototype_cfg.get("visual_alpha", 1.0)),
        top_k=cfg.get("loss", {}).get("top_k", 4),
        use_amp=bool(cfg.get("train", {}).get("use_amp", True)),
    )
