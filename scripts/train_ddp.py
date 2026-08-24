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
    use_mixed_precision: bool,
) -> dict[str, float]:
    model.eval()

    losses = {}

    for split in ["train", "val"]:
        split_losses = []

        for _ in range(eval_iters):
            x, y = dataset.get_batch(split=split, batch_size=batch_size)

            with get_autocast_context(use_mixed_precision):
                _, loss = model(x, y)

            if loss is None:
                raise RuntimeError("Loss should not be None during evaluation.")

            split_losses.append(loss.item())

        losses[split] = sum(split_losses) / len(split_losses)

    model.train()
    return losses


def main() -> None:
    local_rank, rank, world_size, device = setup_ddp()

    try:
        torch.manual_seed(1337 + rank)

        run_name = "tiny_shakespeare_ddp"
        data_path = Path("data/tiny_shakespeare.txt")
        run_dir = Path("runs") / run_name
        checkpoint_dir = Path("checkpoints") / run_name
        latest_checkpoint_path = checkpoint_dir / "latest.pt"
        loss_path = run_dir / "losses.csv"
        config_path = run_dir / "config.json"

        resume = False

        batch_size = 8
        block_size = 32
        max_iters = 500
        eval_interval = 100
        eval_iters = 10
        checkpoint_interval = 100
        learning_rate = 3e-4
        use_mixed_precision = True

        if not data_path.exists():
            raise FileNotFoundError(
                "Dataset not found. Run: python scripts/download_tiny_shakespeare.py"
            )

        dataset = TextTokenDataset(
            TextDatasetConfig(
                input_path=str(data_path),
                block_size=block_size,
                encoding_name="gpt2",
                device=device,
            )
        )

        config = Config(
            vocab_size=dataset.vocab_size,
            block_size=block_size,
            n_embd=64,
            n_head=4,
            n_layer=2,
            dropout=0.1,
            bias=True,
        )

        model = Model(config)
        model = model.to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        scaler = torch.cuda.amp.GradScaler(enabled=use_mixed_precision)

        start_step = 0

        if resume and latest_checkpoint_path.exists():
            start_step = load_checkpoint(
                model=model,
                optimizer=optimizer,
                checkpoint_path=latest_checkpoint_path,
                device=device,
            )
            if is_main_process(rank):
                print(f"Resumed from checkpoint at step {start_step}")

        ddp_model = DDP(
            model, 
            device_ids=[local_rank],
            output_device=local_rank,
        )

        if is_main_process(rank):
            save_config(config, config_path)
            print(f"DDP world size: {world_size}")
            print(f"rank: {rank}")
            print(f"local rank: {local_rank}")
            print(f"device: {device}")
            print(f"mixed precision: {use_mixed_precision}")
            print(f"dataset: {data_path}")
            print(f"vocab size: {dataset.vocab_size}")
            print(f"parameters: {count_parameters(model):,}")

        for step in range(start_step, max_iters):
            if step % eval_interval == 0:
                losses = estimate_loss(
                    model=ddp_model,
                    dataset=dataset,
                    batch_size=batch_size,
                    eval_iters=eval_iters,
                    use_mixed_precision=use_mixed_precision,
                )

                train_loss = losses["train"]
                val_loss = losses["val"]

                if is_main_process(rank):
                    append_loss_row(
                        output_path=loss_path,
                        step=step,
                        train_loss=train_loss,
                        val_loss=val_loss,
                    )

                    print(
                        f"step {step}: "
                        f"train loss {losses['train']:.4f}, "
                        f"val loss {losses['val']:.4f}"
                    )

            x, y = dataset.get_batch(split="train", batch_size=batch_size)

            with get_autocast_context(use_mixed_precision):
                _, loss = ddp_model(x, y)

            if loss is None:
                raise RuntimeError("Loss should not be None during training.")

            optimizer.zero_grad(set_to_none=True)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if is_main_process(rank) and (step + 1) % checkpoint_interval == 0:
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    step=step,
                    config=config,
                    output_path=latest_checkpoint_path,
                )
                print(f"saved checkpoint: {latest_checkpoint_path}")

        if is_main_process(rank):
            final_checkpoint_path = checkpoint_dir / "final.pt"

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                step=max_iters - 1,
                config=config,
                output_path=final_checkpoint_path,
            )

            print(f"saved final checkpoint: {final_checkpoint_path}")

    finally:
        cleanup_ddp()


if __name__ == "__main__":
    main()
