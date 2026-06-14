import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

from transformer.config import Config


class FeedForward(nn.Module):
    """
    Token-wise feed forward network
    
    Attention lets tokens communicate with each other
    Feed-forward transforms each token independently
    
    Shape:
        input:  (batch, time, n_embd)
        output: (batch, time, n_embd)
    """
    
    def __init__(self, config: Config):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias),
            nn.Dropout(config.dropout),
        )

    def forward(
        self,
        x: Float[Tensor, "batch time channels"],
    ) -> Float[Tensor, "batch time channels"]:
        return self.net(x)