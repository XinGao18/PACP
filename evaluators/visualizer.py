from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


class AnomalyScoreVisualizer:
    def __init__(self, save_dir: str | Path) -> None:
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def plot(self, video_name: str, scores, gt=None) -> Path:
        scores = np.asarray(scores)
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(scores, label="score", linewidth=1.5)
        if gt is not None:
            gt = np.asarray(gt)
            ax.fill_between(np.arange(len(gt)), 0, gt, alpha=0.2, label="gt")
        ax.set_title(video_name)
        ax.set_xlabel("frame")
        ax.set_ylabel("anomaly score")
        ax.legend(loc="best")
        fig.tight_layout()
        path = self.save_dir / f"{safe_name(video_name)}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path


def safe_name(value: str) -> str:
    return value.replace("/", "_").replace("\\", "_").replace(":", "_")
