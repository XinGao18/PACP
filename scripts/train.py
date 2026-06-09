from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from evaluators import Evaluator, build_classification_targets, classification_onehot, compute_accuracy, compute_macro_f1, compute_map
from losses import build_total_loss
from models.pcpl_model import build_pcpl_model
from utils.config import load_config
from utils.feature_dataset import NpyFeatureDataset, collate_feature_batch
from utils.model_utils import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PCPL on frozen CLIP appearance features")
    parser.add_argument("--config", type=str, default="configs/ucf_crime.yaml")
    parser.add_argument("--feature-dir", type=str, required=True)
    parser.add_argument("--eval-feature-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="experiments/train_run")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def train_one_epoch(model, loader, optimizer, loss_fn, device: torch.device) -> Dict[str, float]:
    model.train()
    totals: Dict[str, float] = {}
    for batch in tqdm(loader, desc="train", leave=False):
        features = batch["frames_or_features"].to(device)
        labels = batch["video_label"].to(device)
        outputs = model(features, is_feature_input=True)
        losses = loss_fn(outputs=outputs, labels=labels)
        optimizer.zero_grad(set_to_none=True)
        losses["loss"].backward()
        optimizer.step()
        for key, value in losses.items():
            if torch.is_tensor(value) and value.ndim == 0:
                totals[key] = totals.get(key, 0.0) + float(value.detach().cpu())
    return {key: value / max(1, len(loader)) for key, value in totals.items()}


@torch.inference_mode()
def validate(model, loader, loss_fn, device: torch.device, anomaly_score: str) -> Dict[str, float]:
    model.eval()
    totals: Dict[str, float] = {}
    for batch in loader:
        features = batch["frames_or_features"].to(device)
        labels = batch["video_label"].to(device)
        outputs = model(features, is_feature_input=True)
        losses = loss_fn(outputs=outputs, labels=labels)
        for key, value in losses.items():
            if torch.is_tensor(value) and value.ndim == 0:
                totals[f"val_{key}"] = totals.get(f"val_{key}", 0.0) + float(value.detach().cpu())
    metrics = {key: value / max(1, len(loader)) for key, value in totals.items()}
    predictions = Evaluator(model, loader, device=device, anomaly_score=anomaly_score).predict(interpolate_to_frames=False)
    metrics.update({f"val_{key}": value for key, value in prediction_metrics(predictions, len(model.class_names)).items()})
    return metrics


def best_metric(metrics: Dict[str, float], primary_metric: str) -> tuple[float, bool, bool]:
    metric_name = primary_metric.lower()
    metric_name = {"map": "classification_map", "mAP": "classification_map", "f1": "macro_f1"}.get(metric_name, metric_name)
    if metric_name in {"loss", "val_loss"}:
        return float(metrics.get("val_loss", metrics["loss"])), False, True

    value = metrics.get(f"val_{metric_name}")
    if value is not None and math.isfinite(float(value)):
        return float(value), True, True
    fallback = metrics.get("val_loss", metrics["loss"])
    return float(fallback), False, False


def prediction_metrics(predictions: list[dict], num_classes: int) -> Dict[str, float]:
    cls_true, cls_pred, cls_score = build_classification_targets(predictions, num_classes=num_classes)
    return {
        "accuracy": compute_accuracy(cls_true, cls_pred),
        "macro_f1": compute_macro_f1(cls_true, cls_pred),
        "classification_map": compute_map(classification_onehot(cls_true, num_classes), cls_score),
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    train_cfg = cfg.get("train", {})
    dataset_cfg = cfg.get("dataset", {})
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed if args.seed is not None else int(cfg.get("project", {}).get("seed", 42)))

    feature_key = dataset_cfg.get("feature_key", "appearance")
    train_dataset = NpyFeatureDataset(args.feature_dir, feature_key=feature_key)
    val_dataset = NpyFeatureDataset(args.eval_feature_dir, feature_key=feature_key) if args.eval_feature_dir else None
    batch_size = args.batch_size or int(train_cfg.get("batch_size", 8))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_feature_batch)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_feature_batch) if val_dataset else None

    model = build_pcpl_model(cfg).to(device)
    loss_fn = build_total_loss(cfg)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.lr or float(train_cfg.get("lr", 1e-4)),
        weight_decay=args.weight_decay or float(train_cfg.get("weight_decay", 1e-4)),
    )
    epochs = args.epochs or int(train_cfg.get("epochs", 50))
    best_value = None
    best_uses_primary = False
    primary_metric = str(cfg.get("metrics", {}).get("primary", "loss"))
    history = []
    anomaly_score = cfg.get("metrics", {}).get("anomaly_score", "one_minus_normal")
    for epoch in range(epochs):
        metrics = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        if val_loader is not None:
            metrics.update(validate(model, val_loader, loss_fn, device, anomaly_score))
        history.append({"epoch": epoch + 1, **metrics})
        print(f"epoch {epoch + 1}/{epochs}: {metrics}")
        state = {"model": model.state_dict(), "cfg": dict(cfg), "epoch": epoch + 1, "class_names": model.class_names}
        torch.save(state, output_dir / "latest.pt")
        current_value, uses_primary, primary_available = best_metric(metrics, primary_metric)
        should_save_best = best_value is None or (
            primary_available and uses_primary and not best_uses_primary
        ) or (
            uses_primary == best_uses_primary and (current_value > best_value if uses_primary else current_value < best_value)
        )
        if should_save_best:
            best_value = current_value
            best_uses_primary = uses_primary
            torch.save(state, output_dir / "best.pt")

    with (output_dir / "history.json").open("w", encoding="utf-8") as handle:
        json.dump(history, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
