import torch

from transformer.block import FeedForward, LayerNorm
from transformer.config import Config


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

    layer_norm = LayerNorm(config)
    feed_forward = FeedForward(config)

    x = torch.randn(2, 8, 32)

    normalized = layer_norm(x)
    transformed = feed_forward(normalized)

    # This is the residual pattern:
    # keep the original x, then add the transformed output back to it.
    residual_out = x + transformed

    print("input shape:", x.shape)
    print("normalized shape:", normalized.shape)
    print("transformed shape:", transformed.shape)
    print("residual output shape:", residual_out.shape)

    print("normalized mean over channels:", normalized.mean(dim=-1).mean().item())
    print("normalized variance over channels:", normalized.var(dim=-1, unbiased=False).mean().item())


if __name__ == "__main__":
    main()
    