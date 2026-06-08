import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from transformer.config import Config


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
        # TODO: implement

    
    def forward(
            self, 
            x: Float[Tensor, "batch time channels"]
    ) -> Float[Tensor, "batch time head_size"]:
        # TODO: implement
        pass