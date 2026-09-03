import torch

from transformer.attention import MultiHeadAttention
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

    attention = MultiHeadAttention(config)

    x = torch.randn(2, 8, 32)
    out = attention(x)

    print("input shape:", x.shape)
    print("output shape:", out.shape)
    print("n_head:", config.n_head)
    print("head_size:", config.head_size)
    print("n_head * head_size:", config.n_head * config.head_size)


if __name__ == "__main__":
    main()