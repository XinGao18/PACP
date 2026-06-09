from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from evaluators import AnomalyScoreVisualizer, Evaluator, build_classification_targets, classification_onehot, compute_accuracy, compute_macro_f1, compute_map, official_frame_metrics
from models.pcpl_model import build_pcpl_model
from utils.config import load_config
from utils.feature_dataset import NpyFeatureDataset, collate_feature_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PCPL checkpoint")
    parser.add_argument("--config", type=str, default="configs/ucf_crime.yaml")
    parser.add_argument("--feature-dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="experiments/eval")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--smooth", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_key = cfg.get("dataset", {}).get("feature_key", "appearance")
    dataset = NpyFeatureDataset(args.feature_dir, feature_key=feature_key)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_feature_batch)
    model = build_pcpl_model(cfg).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"])

    anomaly_score = cfg.get("metrics", {}).get("anomaly_score", "one_minus_normal")
    predictions = Evaluator(model, loader, device=device, anomaly_score=anomaly_score).predict(interpolate_to_frames=True, smooth=args.smooth)
    cls_true, cls_pred, cls_score = build_classification_targets(predictions, num_classes=len(model.class_names))
    annotation_file = cfg.get("dataset", {}).get("annotation_file")
    if not annotation_file:
        raise ValueError("dataset.annotation_file is required to compute official auc/ap metrics")
    results = {
        **official_frame_metrics(predictions, annotation_file, cfg.get("dataset", {}).get("name", "")),
        "accuracy": compute_accuracy(cls_true, cls_pred),
        "macro_f1": compute_macro_f1(cls_true, cls_pred),
        "classification_map": compute_map(classification_onehot(cls_true, len(model.class_names)), cls_score),
        "num_videos": len(predictions),
    }
    serializable_predictions = []
    for item in predictions:
        class_index = item["predicted_class_index"]
        serializable_predictions.append(
            {
                "video_id": item["video_id"],
                "label": item["label"],
                "class_name": item["class_name"],
                "score": float(np.max(item["segment_scores"])),
                "segment_scores": item["segment_scores"].tolist(),
                "frame_scores": item["frame_scores"].tolist(),
                "class_scores": item["class_scores"].tolist(),
                "predicted_class_index": class_index,
                "predicted_class": model.class_names[class_index],
            }
        )
    with (output_dir / "results.json").open("w", encoding="utf-8") as handle:
        json.dump({"metrics": results, "predictions": serializable_predictions}, handle, ensure_ascii=False, indent=2)
    if args.visualize:
        visualizer = AnomalyScoreVisualizer(output_dir / "vis")
        for item in predictions[:50]:
            visualizer.plot(item["video_id"], item["segment_scores"])
    print(results)


if __name__ == "__main__":
    main()
