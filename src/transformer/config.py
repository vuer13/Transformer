from dataclasses import dataclass
from numbers import Number


@dataclass
class Config:
    """Stores major hyperparameters as attributes and performs basic validation."""

    vocab_size: int         # Number of unique tokens model can predict
    block_size: int         # Maximum context length for predictions    
    n_embd: int             # size of each token embedding vector
    n_head: int             # number of attention heads per Transformer block
    n_layer: int            # number of Transformer blocks
    dropout: float = 0.1    # dropout rate for regularization
    bias: bool = False      # whether to include bias terms in linear layers

    def __post_init__(self):
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be a positive integer")
        
        if self.block_size <= 0:
            raise ValueError("block_size must be a positive integer")
        
        if self.n_embd <= 0:
            raise ValueError("n_embd must be positive")

        if self.n_head <= 0:
            raise ValueError("n_head must be positive")

        if self.n_layer <= 0:
            raise ValueError("n_layer must be positive")

        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")

        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0.0, 1.0)")

    @property
    def head_size(self) -> int:
        """Returns size of each individual attention head."""
        return self.n_embd // self.n_head
