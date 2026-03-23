from __future__ import annotations

import os
from contextlib import nullcontext

import torch
import torch.distributed as dist


def distributed_context() -> dict[str, int | bool]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return {
        "world_size": world_size,
        "rank": rank,
        "local_rank": local_rank,
        "distributed": world_size > 1,
    }


def setup_distributed(device: torch.device) -> dict[str, int | bool]:
    context = distributed_context()
    if not context["distributed"]:
        return context

    backend = "nccl" if device.type == "cuda" else "gloo"
    if device.type == "cuda":
        torch.cuda.set_device(int(context["local_rank"]))
    dist.init_process_group(backend=backend, rank=int(context["rank"]), world_size=int(context["world_size"]))
    return context


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    return rank == 0


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def reduce_scalar(value: float, device: torch.device) -> float:
    if not dist.is_available() or not dist.is_initialized():
        return value
    tensor = torch.tensor(value, dtype=torch.float32, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= dist.get_world_size()
    return float(tensor.item())


def reduce_counts(correct: int, total: int, device: torch.device) -> tuple[int, int]:
    if not dist.is_available() or not dist.is_initialized():
        return correct, total

    pair = torch.tensor([correct, total], dtype=torch.long, device=device)
    dist.all_reduce(pair, op=dist.ReduceOp.SUM)
    return int(pair[0].item()), int(pair[1].item())


def autocast_context(device: torch.device, enabled: bool):
    if device.type != "cuda":
        return nullcontext()
    return torch.cuda.amp.autocast(enabled=enabled)
