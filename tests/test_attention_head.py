import torch

from transformer.attention import SelfAttentionHead
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

    head = SelfAttentionHead(config)

    x = torch.randn(2, 8, 32)
    out = head(x)

    print("input shape:", x.shape)
    print("output shape:", out.shape)
    print("head size:", config.head_size)


if __name__ == "__main__":
    main()