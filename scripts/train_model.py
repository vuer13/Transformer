from pathlib import Path

import torch

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


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"

    if torch.backends.mps.is_available():
        return "mps"

    return "cpu"


@torch.no_grad()
def estimate_loss(
    model: Model,
    dataset: TextTokenDataset,
    batch_size: int,
    eval_iters: int,
) -> dict[str, float]:
    model.eval()

    losses = {}

    for split in ["train", "val"]:
        split_losses = []

        for _ in range(eval_iters):
            x, y = dataset.get_batch(split=split, batch_size=batch_size)
            _, loss = model(x, y)

            if loss is None:
                raise RuntimeError("Loss should not be None during evaluation.")

            split_losses.append(loss.item())

        losses[split] = sum(split_losses) / len(split_losses)

    model.train()
    return losses


def main() -> None:
    torch.manual_seed(1337)

    device = get_device()

    run_name = "tiny_shakespeare"
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

    save_config(config, config_path)

    start_step = 0

    if resume and latest_checkpoint_path.exists():
        start_step = load_checkpoint(
            model=model,
            optimizer=optimizer,
            checkpoint_path=latest_checkpoint_path,
            device=device,
        )
        print(f"Resumed from checkpoint at step {start_step}")

    print(f"device: {device}")
    print(f"dataset: {data_path}")
    print(f"vocab size: {dataset.vocab_size}")
    print(f"parameters: {count_parameters(model):,}")

    for step in range(start_step, max_iters):
        if step % eval_interval == 0:
            losses = estimate_loss(
                model=model,
                dataset=dataset,
                batch_size=batch_size,
                eval_iters=eval_iters,
            )

            train_loss = losses["train"]
            val_loss = losses["val"]

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

        _, loss = model(x, y)

        if loss is None:
            raise RuntimeError("Loss should not be None during training.")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if (step + 1) % checkpoint_interval == 0:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                step=step,
                config=config,
                output_path=latest_checkpoint_path,
            )
            print(f"saved checkpoint: {latest_checkpoint_path}")

    final_checkpoint_path = checkpoint_dir / "final.pt"

    save_checkpoint(
        model=model,
        optimizer=optimizer,
        step=max_iters - 1,
        config=config,
        output_path=final_checkpoint_path,
    )

    print(f"saved final checkpoint: {final_checkpoint_path}")

    model.eval()

    start = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated = model.generate(start, max_new_tokens=100)

    print("\nGenerated text:")
    print(dataset.decode(generated[0].tolist()))


if __name__ == "__main__":
    main()
