from __future__ import annotations

from typing import Any, Dict, Iterable, List

import numpy as np
import torch


class Evaluator:
    def __init__(
        self,
        model: torch.nn.Module,
        dataloader: Iterable[Dict[str, Any]],
        device: torch.device | str = "cuda",
        anomaly_score: str = "one_minus_normal",
    ) -> None:
        self.model = model.to(device)
        self.dataloader = dataloader
        self.device = torch.device(device)
        self.anomaly_score = anomaly_score

    @torch.inference_mode()
    def predict(self, interpolate_to_frames: bool = True, smooth: bool = False) -> List[Dict[str, Any]]:
        self.model.eval()
        predictions: List[Dict[str, Any]] = []
        for batch in self.dataloader:
            features = batch["frames_or_features"].to(self.device)
            outputs = self.model(features, is_feature_input=True)
            segment_probabilities = outputs["class_probabilities"].detach().cpu()
            video_probabilities = outputs["video_probabilities"].detach().cpu()
            anomaly_scores = segment_anomaly_scores(segment_probabilities, self.anomaly_score)
            if smooth:
                anomaly_scores = smooth_scores(anomaly_scores)
            for index, video_id in enumerate(batch["video_id"]):
                segment_scores = anomaly_scores[index].float().numpy()
                frame_count = int(batch["num_frames"][index].item()) if interpolate_to_frames else len(segment_scores)
                class_scores = video_probabilities[index].float().numpy()
                predictions.append(
                    {
                        "video_id": video_id,
                        "label": int(batch["video_label"][index].item()),
                        "class_name": batch["class_name"][index],
                        "segment_scores": segment_scores,
                        "frame_scores": interpolate_scores(segment_scores, frame_count) if interpolate_to_frames else segment_scores,
                        "class_scores": class_scores,
                        "predicted_class_index": int(np.argmax(class_scores)),
                    }
                )
        return predictions


def segment_anomaly_scores(class_probabilities: torch.Tensor, mode: str = "one_minus_normal") -> torch.Tensor:
    if mode == "one_minus_normal":
        return 1.0 - class_probabilities[..., 0]
    if mode == "max_anomaly":
        return class_probabilities[..., 1:].amax(dim=-1)
    raise ValueError(f"Unsupported anomaly score mode: {mode}")


def interpolate_scores(segment_scores: np.ndarray, frame_count: int) -> np.ndarray:
    if frame_count <= 0 or len(segment_scores) == frame_count:
        return segment_scores.astype(np.float32)
    return np.interp(np.arange(frame_count), np.linspace(0, frame_count - 1, num=len(segment_scores)), segment_scores).astype(np.float32)


def smooth_scores(scores: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    if kernel_size <= 1 or scores.shape[1] < 2:
        return scores
    weight = torch.ones(1, 1, kernel_size, device=scores.device, dtype=scores.dtype) / kernel_size
    return torch.nn.functional.conv1d(scores.unsqueeze(1), weight, padding=kernel_size // 2).squeeze(1)
