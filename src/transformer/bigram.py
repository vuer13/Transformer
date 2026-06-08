import torch
import torch.nn
import torch.nn.functional as F
from jaxtyping import Int, Float
from torch import Tensor


class BigramLanguageModel(torch.nn.Module):
    """
    Simple Bigram language model
    
    Model predicts next token using only the current token.

    Purpose: Sanity check before building more complex transformer model.
    """

    def __init__(self, vocab_size: int, n_embd: int = 128) -> None:
        super().__init__()

        # Each token directly maps to logits over the vocabulary
        # Scores for every possible next token in embedding table
        self.token_embedding_table = torch.nn.Embedding(
            num_embeddings=vocab_size, 
            embedding_dim=n_embd,
        )

        self.lm_head = torch.nn.Linear(n_embd, vocab_size)

    def forward(
        self,
        idx: Int[Tensor, "batch time"],
        targets: Int[Tensor, "batch time"] | None = None
    ) -> tuple[Float[Tensor, "batch time vocab"], Tensor | None]:
        """
        Forward pass

        Args:
            idx: (B, T) tensor of token indices
            targets: (B, T) tensor of target token indices

        Returns:
            logits: (B, T, C) tensor of logits for next token prediction
            loss: Cross-entropy loss if targets is provided, else None
        """
        # idx shape: (B, T)
        # logits shape: (B, T, C) where C is vocab size

        token_embd = self.token_embedding_table(idx)
        logits = self.lm_head(token_embd)
        loss = None

        if targets is not None:
            B, T, C = logits.shape

            # PyTorch's cross-entropy loss expects input of shape (B*T, C) and target of shape (B*T)
            logits_flat = logits.view(B * T, C)
            targets_flat = targets.view(B * T)

            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss
    
    @torch.no_grad()
    def generate(
        self,
        idx: Int[Tensor, "batch time"],
        max_new_tokens: int
    ) -> Int[Tensor, "batch generated_time"]:
        """
        Generate new tokens autoregressively

        Steps:
        1. Get logits from model
        2. Take logits from last time step
        3. Apply softmax to get probabilities
        4. Sample from distribution to get next token
        5. Append sampled token to input and repeat
        """

        for _ in range(max_new_tokens):
            logits, _ = self(idx)  # (B, T, C)

            # Only use final time step
            last_logits = logits[:, -1, :]  # (B, C)

            probs = F.softmax(last_logits, dim=-1)  # (B, C)

            # Sample one next token from probability distribution
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)

            # Append sampled token to running sequence
            idx = torch.cat((idx, next_token), dim=1)  # (B, T+1)

        return idx