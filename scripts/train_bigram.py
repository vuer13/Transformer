import torch

from transformer.bigram import BigramLanguageModel
from transformer.data import TextDatasetConfig, TextTokenDataset


def main() -> None:
    # Basic configuration
    batch_size = 32
    block_size = 8
    max_iters = 500
    learning_rate = 1e-2
    eval_interval = 100

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load tokenized dataset
    dataset = TextTokenDataset(
        TextDatasetConfig(
            input_path="data/input.txt",
            block_size=block_size,
            encoding_name="gpt2",
            device=device
        )
    )

    # Initialize model
    model = BigramLanguageModel(vocab_size=dataset.vocab_size, n_embd=128)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for step in range(max_iters):
        x, y = dataset.get_batch(split="train", batch_size=batch_size)
        logits, loss = model(x, y)

        if loss is None:
            raise RuntimeError("Loss should not be None during training")
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % eval_interval == 0:
            print(f"Step {step}: Train Loss = {loss.item():.4f}")

    # Generate text after single starting token
    start = torch.zeros((1, 1), dtype=torch.long, device=device)  # Start with token index 0
    generated = model.generate(start, max_new_tokens=100)

    print("\nGenerated token indices:")
    print(dataset.decode(generated[0].tolist()))


if __name__ == "__main__":
    main()