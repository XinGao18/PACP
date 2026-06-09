from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from tqdm import tqdm

from data.data_factory import build_dataset
from models.encoders import CLIPImageEncoder
from utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract frozen CLIP appearance features")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def dataset_kwargs(cfg: Dict[str, Any]) -> Dict[str, Any]:
    dataset_cfg = dict(cfg.get("dataset", {}))
    dataset_cfg.pop("name", None)
    dataset_cfg["feature_path"] = None
    return dataset_cfg


def output_dir(cfg: Dict[str, Any], output: str | None) -> Path:
    if output:
        return Path(output)
    dataset_cfg = cfg.get("dataset", {})
    return Path("experiments") / dataset_cfg.get("name", "dataset") / dataset_cfg.get("split", "train")


def build_encoder(cfg: Dict[str, Any], device: torch.device) -> CLIPImageEncoder:
    model_cfg = cfg.get("model", {})
    encoder = CLIPImageEncoder(
        model_name=model_cfg.get("clip_model", "ViT-B-16"),
        device=device,
        use_amp=bool(cfg.get("train", {}).get("use_amp", True)),
        download_root=model_cfg.get("clip_download_root", "models/clip"),
    ).to(device)
    encoder.eval()
    return encoder


@torch.inference_mode()
def encode_frames(encoder: CLIPImageEncoder, frames: torch.Tensor, device: torch.device) -> np.ndarray:
    features = encoder(frames.unsqueeze(0).to(device, non_blocking=True))
    return features.squeeze(0).cpu().numpy()


def safe_name(value: str) -> str:
    return value.replace("/", "_").replace("\\", "_")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device(args.device)
    dataset = build_dataset(cfg.get("dataset", {}).get("name", "ucf_crime"), dataset_kwargs(cfg))
    encoder = build_encoder(cfg, device)
    save_dir = output_dir(cfg, args.output)
    save_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = save_dir / "manifest.json"
    existing = set()
    if args.resume and manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as handle:
            existing = {entry["file"] for entry in json.load(handle)}

    manifest = []
    for item in tqdm(dataset, desc="Extracting appearance features"):
        video_id = item["video_id"]
        filename = f"{safe_name(video_id)}.npy"
        if args.resume and filename in existing and (save_dir / filename).exists():
            manifest.append({"video_id": video_id, "video_label": int(item["video_label"]), "class_name": item["class_name"], "file": filename})
            continue
        payload = {
            "video_id": video_id,
            "video_label": int(item["video_label"]),
            "class_name": item["class_name"],
            "num_frames": int(item["num_frames"]),
            "frame_indices": item["frame_indices"].numpy(),
            "appearance": encode_frames(encoder, item["frames_or_features"], device),
        }
        np.save(save_dir / filename, payload, allow_pickle=True)
        manifest.append({"video_id": video_id, "video_label": payload["video_label"], "class_name": payload["class_name"], "file": filename})

    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    print(f"Saved {len(manifest)} feature files to {save_dir}")


if __name__ == "__main__":
    main()
