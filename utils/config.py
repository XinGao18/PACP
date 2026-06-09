from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import yaml


class Config(dict):
    """Lightweight dict-like config wrapper."""


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(config_path: str) -> Config:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    base_ref = cfg.pop("_base_", None)
    if base_ref:
        base_path = (path.parent / base_ref).resolve()
        base_cfg = load_config(str(base_path))
        cfg = _deep_merge(base_cfg, cfg)

    return Config(cfg)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PCPL config loader")
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()