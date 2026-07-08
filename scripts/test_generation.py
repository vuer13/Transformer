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
    model.eval()

    # Start with one token.
    start = torch.zeros((1, 1), dtype=torch.long)

    generated = model.generate(start, max_new_tokens=20)

    print("start shape:", start.shape)
    print("generated shape:", generated.shape)
    print("generated token IDs:", generated)


if __name__ == "__main__":
    main()