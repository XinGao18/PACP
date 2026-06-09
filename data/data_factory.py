from __future__ import annotations

from typing import Any, Dict, Optional

from torch.utils.data import DataLoader


# Deferred import mapping keeps this module lightweight and avoids circular dependencies.
def build_dataloader(
    dataset_name: str,
    dataset_kwargs: Optional[Dict[str, Any]] = None,
    loader_kwargs: Optional[Dict[str, Any]] = None,
) -> DataLoader:
    dataset = build_dataset(dataset_name, dataset_kwargs or {})
    return DataLoader(dataset, **(loader_kwargs or {}))


def build_dataset(dataset_name: str, dataset_kwargs: Optional[Dict[str, Any]] = None):
    name = dataset_name.lower().replace("-", "_")
    kwargs = dataset_kwargs or {}
    if name in {"ucf", "ucf_crime", "ucfcrime"}:
        from data.datasets.ucf_crime import UCFCrimeDataset

        return UCFCrimeDataset(**kwargs)
    if name in {"xd", "xd_violence", "xdviolence"}:
        from data.datasets.xd_violence import XDViolenceDataset

        return XDViolenceDataset(**kwargs)
    raise ValueError(f"Unsupported dataset: {dataset_name}")
