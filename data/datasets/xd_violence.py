from __future__ import annotations

import re

from data.datasets.base_dataset import BaseVideoDataset, VideoSample, split_entry_to_video_id


class XDViolenceDataset(BaseVideoDataset):
    @property
    def dataset_prefix(self) -> str:
        return "xd"

    def _sample_from_split_entry(self, entry: str) -> VideoSample:
        video_id = split_entry_to_video_id(entry)
        class_name = self._class_from_entry(entry, video_id)
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
        for token in tokens[1:]:
            if token.lower() in self.class_to_index and token.lower() != "normal":
                return self.canonical_class_name(token)
        for class_name in self.class_names[1:]:
            if re.search(rf"(?<![A-Za-z0-9]){re.escape(class_name)}(?![A-Za-z0-9])", video_id):
                return class_name
        return "normal"
