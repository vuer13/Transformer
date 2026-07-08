import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor

from transformer.block import LayerNorm, TransformerBlock
from transformer.config import Config


class Model(nn.Module):
    """
    Full decoder-only language model

    Sequence:
    token IDs -> token embeddings
    -> positional embeddings -> Transformer blocks
    -> final LayerNorm -> language modeling head
    -> logits over vocabulary
    """

    def __init__(self, config: Config):
        super().__init__()

        self.config = config

        self.token_embedding_table = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.n_embd,
        )
        self.position_embedding_table = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.n_embd,
        )
        self.blocks = nn.ModuleList([
            TransformerBlock(config)
            for _ in range(config.n_layer)
        ])
        self.ln_final = LayerNorm(config)
        self.lm_head = nn.Linear(
            in_features=config.n_embd,
            out_features=config.vocab_size,
            bias=config.bias,
        )

    def forward(
        self,
        idx: Int[Tensor, "batch time"],
        targets: Int[Tensor, "batch time"] | None = None,
    ) -> tuple[Float[Tensor, "batch time vocab"], Tensor | None]:
        """
        Args:
            idx: Token IDs with shape (B, T)
            targets: Optional next-token targets with shape (B, T)

        Returns:
            logits: shape (B, T, vocab_size)
            loss: if targets is provided, else None (cross-entropy)
        """
        B, T = idx.shape

        if T > self.config.block_size:
            raise ValueError(
                f"Sequence length T={T} exceeds block size {self.config.block_size}"
            )
        
        # Token embeddings
        # idx: (B, T)
        # tok_embd: (B, T, n_embd)
        tok_embd = self.token_embedding_table(idx)

        # Position embeddings:
        # positions: (T)
        # pos_embd: (T, n_embd)
        positions = torch.arange(T, device=idx.device)
        pos_embd = self.position_embedding_table(positions)

        # Broadcast position embeddings aacross the batch
        # x shape: (B, T, n_embd)
        x = tok_embd + pos_embd

        for block in self.blocks:
            x = block(x)

        x = self.ln_final(x)

        # logits shape: (B, T, vocab_size)
        logits = self.lm_head(x)

        loss = None

        if targets is not None:
            B, T, C = logits.shape

            # Cross entropy expects
            # logits: (B*T, vocab_size)
            # targets: (B*T)
            logits_flat = logits.reshape(B * T, C)
            targets_flat = targets.reshape(B * T)

            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss
