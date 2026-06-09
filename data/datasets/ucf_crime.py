from __future__ import annotations

import re

from data.datasets.base_dataset import BaseVideoDataset, VideoSample, is_normal_class, split_entry_to_video_id, strip_video_suffix


class UCFCrimeDataset(BaseVideoDataset):
    @property
    def dataset_prefix(self) -> str:
        return "ucf"

    def _sample_from_split_entry(self, entry: str) -> VideoSample:
        video_id = split_entry_to_video_id(entry)
        class_name = self.canonical_class_name(self._class_from_entry(entry, video_id))
        annotation = self._annotation_for(video_id)
        return VideoSample(
            video_id=video_id,
            video_label=self.class_index(class_name),
            class_name=class_name,
            frame_dir=self._frame_dir_for(entry),
            num_frames_hint=annotation.get("num_frames_hint"),
        )

    def _class_from_entry(self, entry: str, video_id: str) -> str:
        tokens = entry.split()
        if len(tokens) > 1 and not tokens[1].lstrip("-").isdigit():
            return tokens[1]
        parent = tokens[0].replace("\\", "/").split("/")[-2] if "/" in tokens[0].replace("\\", "/") else ""
        if parent and not is_normal_class(parent):
            return parent
        if is_normal_class(video_id):
            return "normal"
        match = re.match(r"[A-Za-z_]+", strip_video_suffix(video_id))
        return match.group(0).rstrip("_") if match else "normal"
