from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score

VIDEO_SUFFIXES = (".mp4", ".avi", ".mkv", ".mov", ".webm")


def compute_auc(y_true, y_score) -> float:
    y_true = np.asarray(y_true).reshape(-1)
    y_score = np.asarray(y_score).reshape(-1)
    if y_true.size == 0 or len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def compute_ap(y_true, y_score) -> float:
    y_true = np.asarray(y_true).reshape(-1)
    y_score = np.asarray(y_score).reshape(-1)
    if y_true.size == 0:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def compute_accuracy(y_true, y_pred) -> float:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    if y_true.size == 0:
        return float("nan")
    return float(accuracy_score(y_true, y_pred))


def compute_macro_f1(y_true, y_pred) -> float:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    if y_true.size == 0:
        return float("nan")
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def compute_map(y_true, y_score) -> float:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if y_true.ndim != 2 or y_score.ndim != 2 or y_true.shape != y_score.shape or y_true.shape[1] == 0:
        return float("nan")
    values = []
    for index in range(y_true.shape[1]):
        if len(np.unique(y_true[:, index])) < 2:
            continue
        values.append(average_precision_score(y_true[:, index], y_score[:, index]))
    return float(np.mean(values)) if values else float("nan")


def build_classification_targets(predictions: list[dict], num_classes: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_true = np.asarray([item["label"] for item in predictions], dtype=np.int64)
    y_score = np.asarray([item["class_scores"] for item in predictions], dtype=np.float32)
    y_pred = np.asarray([item["predicted_class_index"] for item in predictions], dtype=np.int64)
    if num_classes is None:
        num_classes = y_score.shape[1] if y_score.ndim == 2 else int(y_true.max(initial=0) + 1)
    return y_true, y_pred, y_score



def classification_onehot(labels: np.ndarray, num_classes: int) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    onehot = np.zeros((len(labels), num_classes), dtype=np.float32)
    valid = (labels >= 0) & (labels < num_classes)
    onehot[np.arange(len(labels))[valid], labels[valid]] = 1.0
    return onehot


def load_official_annotations(annotation_file: str | Path, dataset_name: str) -> Dict[str, List[Tuple[int, int]]]:
    path = resolve_path(annotation_file)
    if not path.exists():
        raise FileNotFoundError(f"Annotation file not found: {path}")
    annotations: Dict[str, List[Tuple[int, int]]] = {}
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            tokens = line.strip().split()
            if not tokens:
                continue
            video_id = normalize_video_id(tokens[0])
            number_tokens = tokens[2:] if dataset_name.lower().startswith("ucf") else tokens[1:]
            numbers = [int(token) for token in number_tokens if token.lstrip("-").isdigit()]
            segments = [(start, end) for start, end in zip(numbers[0::2], numbers[1::2]) if start >= 0 and end >= start]
            annotations.setdefault(video_id, []).extend(segments)
    return annotations


def official_frame_targets(predictions: Sequence[dict], annotations: Dict[str, List[Tuple[int, int]]]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    all_labels = []
    all_scores = []
    missing = []
    for item in predictions:
        video_id = normalize_video_id(item["video_id"])
        scores = np.asarray(item["frame_scores"], dtype=np.float32)
        labels = np.zeros(len(scores), dtype=np.int64)
        if video_id not in annotations and int(item.get("label", 0)) > 0:
            missing.append(video_id)
        for start, end in annotations.get(video_id, []):
            start = max(0, min(len(labels), start))
            end = max(0, min(len(labels), end + 1))
            labels[start:end] = 1
        all_labels.append(labels)
        all_scores.append(scores)
    if not all_labels:
        return np.asarray([]), np.asarray([]), missing
    return np.concatenate(all_labels), np.concatenate(all_scores), missing


def official_frame_metrics(predictions: Sequence[dict], annotation_file: str | Path, dataset_name: str) -> dict[str, float | int]:
    annotations = load_official_annotations(annotation_file, dataset_name)
    labels, scores, missing = official_frame_targets(predictions, annotations)
    return {
        "auc": compute_auc(labels, scores),
        "ap": compute_ap(labels, scores),
        "official_missing_annotations": len(missing),
        "official_num_frames": int(labels.size),
    }


def resolve_path(path: str | Path) -> Path:
    normalized = str(path).strip().replace("\\", "/")
    candidates = [normalized]
    drive_match = None
    if len(normalized) >= 3 and normalized[1:3] == ":/":
        drive_match = (normalized[0], normalized[3:])
    msys_match = None
    if len(normalized) >= 3 and normalized[0] == "/" and normalized[2] == "/":
        msys_match = (normalized[1], normalized[3:])
    if drive_match:
        drive, rest = drive_match
        candidates.append(f"/{drive.lower()}/{rest}")
    elif msys_match:
        drive, rest = msys_match
        candidates.append(f"{drive.upper()}:/{rest}")
    for candidate in candidates:
        candidate_path = Path(candidate)
        if candidate_path.exists():
            return candidate_path
    return Path(candidates[-1])


def normalize_video_id(video_name: str) -> str:
    value = Path(video_name.replace("\\", "/")).name
    for suffix in VIDEO_SUFFIXES:
        if value.lower().endswith(suffix):
            value = value[: -len(suffix)]
            break
    return value
