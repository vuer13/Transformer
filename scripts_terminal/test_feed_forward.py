import torch

from transformer.block import FeedForward
from transformer.config import Config


def main() -> None:
    config = Config(
        vocab_size=50257,
        block_size=8,
        n_embd=32,
        n_head=4,
        n_layer=2,
        dropout=0.1,
        bias=False,
    )

    feed_forward = FeedForward(config)

    x = torch.randn(2, 8, 32)
    out = feed_forward(x)

    print("input shape:", x.shape)
    print("output shape:", out.shape)


if __name__ == "__main__":
    main()