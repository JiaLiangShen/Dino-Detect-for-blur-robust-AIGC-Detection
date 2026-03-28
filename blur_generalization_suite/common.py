import json
import logging
import os
import random
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.distributed as dist
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


DEFAULT_TIMEOUT = timedelta(minutes=30)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logging(rank: int = 0) -> None:
    level = logging.INFO if rank == 0 else logging.WARNING
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")


def infer_backend(world_size: int) -> str:
    if torch.cuda.is_available() and torch.cuda.device_count() >= world_size:
        return "nccl"
    return "gloo"


def setup_distributed(rank: int, world_size: int, backend: str | None = None) -> None:
    if world_size <= 1 or dist.is_initialized():
        return

    backend = backend or infer_backend(world_size)
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")

    if backend == "nccl":
        os.environ.setdefault("NCCL_TIMEOUT", "1800")
        os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")
        os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
        os.environ.setdefault("NCCL_SOCKET_IFNAME", "^lo,docker")
        os.environ.setdefault("NCCL_IB_DISABLE", "1")
        os.environ.setdefault("NCCL_P2P_DISABLE", "1")
        if not torch.cuda.is_available():
            raise RuntimeError("NCCL backend requires CUDA, but CUDA is unavailable.")
        torch.cuda.set_device(rank)

    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size,
        timeout=DEFAULT_TIMEOUT,
    )


def cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def is_rank_zero(rank: int) -> bool:
    return not dist.is_initialized() or rank == 0


def barrier(world_size: int) -> None:
    if world_size > 1 and dist.is_initialized():
        dist.barrier()


def all_reduce_tensor(tensor: torch.Tensor, world_size: int) -> torch.Tensor:
    if world_size > 1 and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: str | Path, payload: Dict[str, Any]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def remove_module_prefix(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    cleaned = {}
    for key, value in state_dict.items():
        cleaned[key[7:] if key.startswith("module.") else key] = value
    return cleaned


def extract_trainable_state_dict(model: torch.nn.Module) -> Dict[str, Any]:
    state_dict = model.state_dict()
    trainable_names = {name for name, param in model.named_parameters() if param.requires_grad}
    return {
        name: state_dict[name].detach().cpu()
        for name in trainable_names
        if name in state_dict
    }


def load_checkpoint_state(path: str | Path, map_location: str | torch.device = "cpu") -> Tuple[Dict[str, Any], Dict[str, Any]]:
    checkpoint = torch.load(path, map_location=map_location)
    if isinstance(checkpoint, dict):
        if "trainable_state_dict" in checkpoint:
            state_dict = checkpoint["trainable_state_dict"]
        elif "adapter_state_dict" in checkpoint:
            state_dict = checkpoint["adapter_state_dict"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    return checkpoint, remove_module_prefix(state_dict)


def compute_binary_metrics(labels: list[int], predictions: list[int]) -> Dict[str, Any]:
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="binary",
        zero_division=0,
    )
    cm = confusion_matrix(labels, predictions, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    real_total = tn + fp
    fake_total = tp + fn
    real_accuracy = float(tn / real_total) if real_total else 0.0
    fake_accuracy = float(tp / fake_total) if fake_total else 0.0
    balanced_accuracy = 0.5 * (real_accuracy + fake_accuracy)
    balanced_accuracy_half_gap = abs(fake_accuracy - real_accuracy) / 2.0

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "bacc": float(balanced_accuracy),
        "balanced_accuracy": float(balanced_accuracy),
        "balanced_accuracy_half_gap": float(balanced_accuracy_half_gap),
        "real_accuracy": real_accuracy,
        "fake_accuracy": fake_accuracy,
        "real_total": int(real_total),
        "fake_total": int(fake_total),
        "confusion_matrix": cm.tolist(),
        "total_samples": int(len(labels)),
    }


def count_trainable_parameters(model: torch.nn.Module) -> Dict[str, int]:
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    total = sum(param.numel() for param in model.parameters())
    return {"trainable": int(trainable), "total": int(total)}


def parse_distributed_env() -> Tuple[int, int, int]:
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    return rank, world_size, local_rank
