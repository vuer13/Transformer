import os
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from transformer.config import Config
from transformer.data import TextDatasetConfig, TextTokenDataset
from transformer.model import Model
from transformer.training import (
    append_loss_row,
    count_parameters,
    load_checkpoint,
    save_checkpoint,
    save_config,
)


def setup_ddp() -> tuple[int, int, int, str]:
    """
    Initializes Distributed Data Parallel Training

    torchrun starts one Python process per GPU.
    Each process gets a LOCAL_RANK, RANK, and WORLD_SIZE.

    LOCAL_RANK = which GPU this process should use on this machine
    RANK = global process ID
    WORLD_SIZE = total number of processes
    """
    if not torch.cuda.is_available():
        raise RuntimeError("DDP training requires CUDA GPUs")
    
    dist.init_process_group(backend="nccl")

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)

    device = f"cuda:{local_rank}"

    return local_rank, rank, world_size, device


def cleanup_ddp() -> None:
    """Shuts down distirbuted process group"""
    dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    """
    Returns True only for rank 0.
    We only want one process to print logs and save checkpoints.
    """
    return rank == 0


def get_autocast_context(use_mixed_precision: bool):
    """
    Uses CUDA mixed precision when enabled, otherwise runs normally.
    """
    if use_mixed_precision:
        return torch.autocast(device_type="cuda", dtype=torch.float16)

    return nullcontext()


@torch.no_grad()
def estimate_loss(
    model: Model,
    dataset: TextTokenDataset,
    batch_size: int,
    eval_iters: int,
    device: str,
    use_mixed_precision: bool,
) -> dict[str, float]:
    model.eval()

    losses = {}

    for split in ["train", "val"]:
        split_losses = []

        for _ in range(eval_iters):
            x, y = dataset.get_batch(split=split, batch_size=batch_size)

            with get_autocast_context(device, use_mixed_precision):
                _, loss = model(x, y)

            if loss is None:
                raise RuntimeError("Loss should not be None during evaluation.")

            split_losses.append(loss.item())

        losses[split] = sum(split_losses) / len(split_losses)

    model.train()
    return losses
