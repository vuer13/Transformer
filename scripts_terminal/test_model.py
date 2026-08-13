import torch

from transformer.config import Config
from transformer.model import Model


def main() -> None:
    config = Config(
        vocab_size=50257,
        block_size=8,
        n_embd=32,
        n_head=4,
        n_layer=2,
        dropout=0.1,
        bias=True,
    )

    model = Model(config)

    idx = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(2, 8),
    )

    targets = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(2, 8),
    )

    logits, loss = model(idx, targets)

    print("idx shape:", idx.shape)
    print("logits shape:", logits.shape)
    print("loss:", loss.item() if loss is not None else None)


if __name__ == "__main__":
    main()