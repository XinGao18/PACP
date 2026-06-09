from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


class CheckpointManager:
    def __init__(self, save_dir: str) -> None:
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def save(self, state: dict[str, Any], name: str = "latest.pt") -> Path:
        path = self.save_dir / name
        torch.save(state, path)
        return path

    def load(self, name: str = "latest.pt") -> dict[str, Any]:
        path = self.save_dir / name
        return torch.load(path, map_location="cpu")