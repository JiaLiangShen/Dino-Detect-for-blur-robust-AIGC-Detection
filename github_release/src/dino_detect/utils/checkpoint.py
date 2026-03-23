from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch


def save_checkpoint(
    path: str | Path,
    model_state: dict[str, Any],
    optimizer_state: Optional[dict[str, Any]] = None,
    scheduler_state: Optional[dict[str, Any]] = None,
    scaler_state: Optional[dict[str, Any]] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> None:
    payload = {
        "model_state_dict": model_state,
        "optimizer_state_dict": optimizer_state,
        "scheduler_state_dict": scheduler_state,
        "scaler_state_dict": scaler_state,
        "metadata": metadata or {},
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, destination)


def strip_module_prefix(state_dict: dict[str, Any]) -> dict[str, Any]:
    return {
        key[7:] if key.startswith("module.") else key: value
        for key, value in state_dict.items()
    }


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=map_location)
    if "model_state_dict" in checkpoint:
        checkpoint["model_state_dict"] = strip_module_prefix(checkpoint["model_state_dict"])
    return checkpoint
