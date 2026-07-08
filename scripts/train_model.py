import torch

from transformer.config import Config
from transformer.data import TextDatasetConfig, TextTokenDataset
from transformer.model import Model


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

    batch_size = 8
    block_size = 32
    max_iters = 500
    eval_interval = 100
    eval_iters = 10
    learning_rate = 3e-4

    dataset = TextTokenDataset(
        TextDatasetConfig(
            input_path="data/input.txt",
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

    print(f"device: {device}")
    print(f"vocab size: {dataset.vocab_size}")
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")

    for step in range(max_iters):
        if step % eval_interval == 0:
            losses = estimate_loss(
                model=model,
                dataset=dataset,
                batch_size=batch_size,
                eval_iters=eval_iters,
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

    start = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated = model.generate(start, max_new_tokens=100)

    print("\nGenerated text:")
    print(dataset.decode(generated[0].tolist()))


if __name__ == "__main__":
    main()
