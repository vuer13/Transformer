import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

from transformer.config import Config


class LayerNorm(nn.Module):
    """
    Manual LayerNorm implementation

    Shape: 
        input:  (batch, time, n_embd)
        output: (batch, time, n_embd)

    LayerNorm normalizes each token vector across its channel dimension.
    """
    def __init__(self, config: Config, eps: float = 1e-5):
        super().__init__()

        self.eps = eps

        # gamma, starts asa ones so it does not change normalized values initially
        self.weight = nn.Parameter(torch.ones(config.n_embd))

        # beta, starts as zeros so it does not shift values initally
        if config.bias:
            self.bias = nn.Parameter(torch.zeros(config.n_embd))
        else:
            self.register_parameter("bias", None)

    def forward(
        self,
        x: Float[Tensor, "batch time channels"],
    ) -> Float[Tensor, "batch time channels"]:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)

        x_norm = (x-mean) / torch.sqrt(var + self.eps)

        if self.bias is None:
            return self.weight * x_norm

        return self.weight * x_norm + self.bias


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
