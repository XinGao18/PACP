from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from data.transforms import get_clip_transform

PathLike = Union[str, Path]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".webm"}


@dataclass(frozen=True)
class VideoSample:
    video_id: str
    video_label: int
    class_name: str
    frame_dir: Path
    num_frames_hint: Optional[int] = None


class BaseVideoDataset(Dataset, ABC):
    def __init__(
        self,
        root: PathLike,
        split: str = "train",
        class_names: Optional[Sequence[str]] = None,
        num_segments: int = 32,
        image_size: int = 224,
        normalization: str = "clip",
        split_file: Optional[PathLike] = None,
        annotation_file: Optional[PathLike] = None,
        feature_path: Optional[PathLike] = None,
        feature_key: str = "appearance",
        transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
    ) -> None:
        self.root = resolve_path(root)
        self.split = normalize_split(split)
        self.class_names = normalize_class_names(class_names)
        self.class_to_index = {name.lower(): index for index, name in enumerate(self.class_names)}
        self.num_segments = int(num_segments)
        self.feature_path = resolve_path(feature_path) if feature_path else None
        self.feature_key = feature_key
        self.split_file = resolve_path(split_file) if split_file else self.root / f"{self.dataset_prefix}_{self.split}.txt"
        self.annotation_file = resolve_path(annotation_file) if annotation_file else self.root / "annotations" / "test_video_annotations.txt"
        is_train = self.split == "train"
        self.transform = transform or get_clip_transform(image_size=image_size, is_train=is_train, normalization=normalization)
        self.annotation_index = self._load_annotation_index()
        self.samples = [self._sample_from_split_entry(entry) for entry in self._read_split_entries()]
        if self.feature_path is None:
            self.samples = [sample for sample in self.samples if sample.frame_dir.exists()]

    @property
    @abstractmethod
    def dataset_prefix(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def _sample_from_split_entry(self, entry: str) -> VideoSample:
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self.samples[index]
        if self.feature_path is not None:
            frames_or_features, original_length = self._load_feature(sample.video_id)
            frame_indices = uniform_indices(original_length, self.num_segments)
            num_frames = sample.num_frames_hint or original_length
        else:
            frame_paths = sorted_image_files(sample.frame_dir)
            if not frame_paths:
                raise FileNotFoundError(f"No image frames found in {sample.frame_dir}")
            frame_indices = uniform_indices(len(frame_paths), self.num_segments)
            frames_or_features = self._load_frames([frame_paths[i] for i in frame_indices])
            num_frames = sample.num_frames_hint or len(frame_paths)

        return {
            "frames_or_features": frames_or_features,
            "video_label": torch.tensor(sample.video_label, dtype=torch.long),
            "class_name": sample.class_name,
            "video_id": sample.video_id,
            "num_frames": torch.tensor(num_frames, dtype=torch.long),
            "frame_indices": torch.tensor(frame_indices, dtype=torch.long),
        }

    def class_index(self, class_name: str) -> int:
        canonical = self.canonical_class_name(class_name)
        key = canonical.lower()
        if key not in self.class_to_index:
            raise ValueError(f"Class {class_name!r} is not in class_names={self.class_names}")
        return self.class_to_index[key]

    def canonical_class_name(self, class_name: str) -> str:
        value = class_name.strip()
        if is_normal_class(value):
            return "normal"
        for name in self.class_names:
            if name.lower() == value.lower():
                return name
        return value

    def _read_split_entries(self) -> List[str]:
        if not self.split_file.exists():
            raise FileNotFoundError(f"Split file not found: {self.split_file}")
        entries: List[str] = []
        with self.split_file.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                stripped = line.strip()
                if stripped and not stripped.startswith("#"):
                    entries.append(stripped)
        return entries

    def _load_annotation_index(self) -> Dict[str, Dict[str, Any]]:
        if not self.annotation_file.exists():
            return {}
        index: Dict[str, Dict[str, Any]] = {}
        with self.annotation_file.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                parsed = parse_annotation_line(line.strip())
                if parsed is not None:
                    index[parsed["video_id"]] = parsed
        return index

    def _annotation_for(self, video_id: str) -> Dict[str, Any]:
        return self.annotation_index.get(video_id, {})

    def _frame_dir_for(self, entry: str) -> Path:
        video_id = split_entry_to_video_id(entry)
        stem = strip_video_suffix(video_id)
        candidates = [self.root / "frames" / video_id, self.root / "frames" / stem, self.root / video_id, self.root / stem]
        for candidate in candidates:
            if candidate.is_dir():
                return candidate
        return candidates[1]

    def _load_frames(self, paths: Sequence[Path]) -> torch.Tensor:
        frames = []
        for path in paths:
            with Image.open(path) as image:
                frames.append(self.transform(image.convert("RGB")))
        return torch.stack(frames, dim=0)

    def _load_feature(self, video_id: str) -> Tuple[torch.Tensor, int]:
        if self.feature_path is None:
            raise RuntimeError("feature_path is not configured")
        path = self._feature_path_for(video_id)
        payload = np.load(path, allow_pickle=True)
        if getattr(payload, "shape", None) == () and payload.dtype == object:
            payload = payload.item()
        if isinstance(payload, dict):
            if self.feature_key not in payload:
                raise KeyError(f"Feature key {self.feature_key!r} not found in {path}")
            array = payload[self.feature_key]
        else:
            array = payload
        features = torch.as_tensor(array, dtype=torch.float32)
        if features.ndim != 2:
            raise ValueError(f"Expected feature shape [T, D] in {path}, got {tuple(features.shape)}")
        original_length = int(features.shape[0])
        return resample_sequence(features, self.num_segments), original_length

    def _feature_path_for(self, video_id: str) -> Path:
        if self.feature_path is None:
            raise RuntimeError("feature_path is not configured")
        if self.feature_path.is_file():
            return self.feature_path
        safe_id = safe_feature_name(video_id)
        candidates = [self.feature_path / f"{video_id}.npy", self.feature_path / f"{safe_id}.npy"]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"No feature file found for {video_id} under {self.feature_path}")


def resolve_path(path: PathLike) -> Path:
    normalized = str(path).strip().replace("\\", "/")
    candidates = [normalized]
    drive_match = re.match(r"^([A-Za-z]):/(.*)$", normalized)
    msys_match = re.match(r"^/([A-Za-z])/(.*)$", normalized)
    if drive_match:
        drive, rest = drive_match.groups()
        candidates.append(f"/{drive.lower()}/{rest}")
    elif msys_match:
        drive, rest = msys_match.groups()
        candidates.append(f"{drive.upper()}:/{rest}")
    for candidate in candidates:
        candidate_path = Path(candidate)
        if candidate_path.exists():
            return candidate_path
    return Path(normalized)


def normalize_split(split: str) -> str:
    value = split.lower()
    return "val" if value in {"valid", "validation", "eval"} else value


def normalize_class_names(class_names: Optional[Sequence[str]]) -> Tuple[str, ...]:
    values = [str(name).strip() for name in (class_names or ["normal", "abnormal"]) if str(name).strip()]
    if not values or values[0].lower() != "normal":
        values.insert(0, "normal")
    deduped: List[str] = []
    seen = set()
    for value in values:
        key = value.lower()
        if key not in seen:
            deduped.append("normal" if key == "normal" else value)
            seen.add(key)
    return tuple(deduped)


def is_normal_class(class_name: str) -> bool:
    return class_name.strip().lower().startswith("normal")


def split_entry_to_video_id(entry: str) -> str:
    return strip_video_suffix(entry.split()[0].replace("\\", "/").split("/")[-1])


def strip_video_suffix(video_name: str) -> str:
    path = Path(video_name)
    return path.stem if path.suffix.lower() in VIDEO_EXTENSIONS else path.name


def uniform_indices(num_items: int, num_segments: int) -> List[int]:
    if num_items <= 0:
        return []
    if num_segments <= 0:
        return list(range(num_items))
    if num_segments == 1:
        return [num_items // 2]
    return [min(num_items - 1, round(index * (num_items - 1) / (num_segments - 1))) for index in range(num_segments)]


def resample_sequence(features: torch.Tensor, num_segments: int) -> torch.Tensor:
    if num_segments <= 0 or features.shape[0] == num_segments:
        return features
    indices = torch.tensor(uniform_indices(int(features.shape[0]), num_segments), dtype=torch.long)
    return features.index_select(0, indices)


def sorted_image_files(directory: Path) -> List[Path]:
    return sorted(
        (path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda path: natural_key(path.name),
    )


def natural_key(value: str) -> Tuple[Any, ...]:
    parts = re.split(r"(\d+)", value)
    return tuple(int(part) if part.isdigit() else part.lower() for part in parts)


def parse_annotation_line(line: str) -> Optional[Dict[str, Any]]:
    if not line:
        return None
    tokens = line.split()
    if len(tokens) < 3:
        return None
    video_id = split_entry_to_video_id(tokens[0])
    return {"video_id": video_id}


def safe_feature_name(value: str) -> str:
    return value.replace("/", "_").replace("\\", "_")
