import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from transformer.config import Config
from transformer.rope import apply_rope


class SelfAttentionHead(nn.Module):
    """
    One masked self-attention head. 
    Input shape: (batch, time, n_embd)
    Output shape: (batch, time, head_size)

    - creates keys, queries, values for all time steps in the input
    - computes attention weights
    - applies causal mask
    - softmax, dropout, weighted sum of values
    """
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        # K = what each token contains
        self.key = nn.Linear(
            in_features=config.n_embd,
            out_features=config.head_size,
            bias=config.bias,
        )

        # Q = what each token is looking for
        self.query = nn.Linear(
            in_features=config.n_embd,
            out_features=config.head_size,
            bias=config.bias,
        )

        # V - actual information passed forward
        self.value = nn.Linear(
            in_features=config.n_embd,
            out_features=config.head_size,
            bias=config.bias,
        )

        self.dropout = nn.Dropout(config.dropout)

        # Lower-triangular casuaal mask
        # Shape: (block_size, block_size)
        # Prevents token from attending to future tokens
        self.register_buffer(
            "tril",
            torch.tril(torch.ones(config.block_size, config.block_size))
        )
    
    def forward(
            self, 
            x: Float[Tensor, "batch time channels"]
    ) -> Float[Tensor, "batch time head_size"]:
        """Run masked self-attention
        Args:
            x: (batch, time, n_embd)
        
        Returns:
            out: (batch, time, head_size)
        """
        B, T, C = x.shape

        if C != self.config.n_embd:
            raise ValueError(f"Expected input embedding size {self.config.n_embd}, got {C}")
        
        if T > self.config.block_size:
            raise ValueError(f"Input sequence length {T} exceeds block size {self.config.block_size}")
        
        # Shape
        # k: (B, T, head_size)
        # q: (B, T, head_size)
        k = self.key(x)
        q = self.query(x)
        
        if self.config.use_rope:
            k = apply_rope(k)
            q = apply_rope(q)

        # Compute attention scores
        # q shape: (B, T, head_size)
        # k.transpose(-2, -1) shape: (B, head_size, T)
        # weights shape: (B, T, T)
        #
        # Each token gets a score against every other token
        weights = q @ k.transpose(-2, -1)

        # Scale by sqrt(head_size) to prevent large values
        weights = weights * (self.config.head_size ** -0.5)

        # Apply casual mask
        # tril[:T, :T] shape: (T, T)
        # Keeps only part needed for current sequence length
        # Positions above the diagonal are set to -inf, preventing attention to future tokens
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float("-inf"))

        # Softmax to get attention probabilities
        weights = F.softmax(weights, dim=-1)
        # Drop some attention connections for regularization
        weights = self.dropout(weights)

        # Shape
        # v: (B, T, head_size)
        # out: (B, T, head_size)
        v = self.value(x)
        out = weights @ v

        return out


class MultiHeadAttention(nn.Module):
    """
    Multiple masked self-attention heads running in parallel
    Each SelfAttentionHead outputs: (batch, time, head_size)
    With n_heads heads, concetenating them gives: 
        (batch, time, n_head * head_size)
    Since n_head * head_size = n_embd
    The final output shape is: (batch, time, n_embd)
    Then we apply a projection layer so the heads can mix information.
    """
    
    def __init__(self, config = Config):
        super().__init__()
        
        self.config = config
        self.heads = nn.ModuleList([
            SelfAttentionHead(config)
            for _ in range(config.n_head)
        ])
        self.proj = nn.Linear(
            in_features=config.n_embd,
            out_features=config.n_embd,
            bias=config.bias,
        )
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(
        self,
        x: Float[Tensor, "batch time channels"],
    ) -> Float[Tensor, "batch time channels"]:
        """
        Run all attention heads, concatenate their outputs and projects
        
        Args:
        x: Tensor of shape (B, T, n_embd)
        
        Returns:
            Tensor of shape (B, T, n_embd)
        """
        
        # Each head returns shape: (B, T, head_size)
        # Concatentating across the channel dimension gives:
        # (B, T, n_head * head_size)
        # Since n_head * head_size = n_embd:
        # (B, T, n_embd)
        out = torch.cat(
            [head(x) for head in self.heads],
            dim=-1,
        )
        
        # Mix information from all heads
        out = self.proj(out)
        
        # Apply dropout after projection
        out = self.dropout(out)
        
        return out
