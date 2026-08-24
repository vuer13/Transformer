import torch
from jaxtyping import Float
from torch import Tensor


def rotate_half(x: Float[Tensor, "batch time channels"]) -> Float[Tensor, "batch time channels"]:
    """
    Splits the last dimension into pairs and rotates each pair

    Example:
    [x1, x2, x3, x4] -> [-x2, x1, -x4, x3]

    RoPE uses this rotation to inject position information into query/key vectors.    
    """
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]

    rotated = torch.stack((-x_odd, x_even), dim=-1)

    return rotated.flatten(start_dim=-2)


def apply_rope(x: Float[Tensor, "batch time channels"]) -> Float[Tensor, "batch time channels"]:
    """
    Applies rotary position embeddings to a query or key tensor.

    Input shape:
    (batch, time, channels)

    Output shape:
    (batch, time, channels)

    It rotates the query/key vector based on token position.
    """
    _, time, channels = x.shape

    if channels % 2 != 0:
        raise ValueError("RoPE requires the last dimension to be even")

    # Creates a tensor containing every token position so each token has its
    # own position in sequence
    positions = torch.arange(time, device=x.device, dtype=x.dtype)
    # Creates indices for every pair of embedding dimensions 
    dim_indices = torch.arange(0, channels, 2, device=x.device, dtype=x.dtype)

    # Computes the inverse frequencies used by RoPE so each pair gets
    # its own rotation frequency
    inv_freq = 1.0 / (10000 ** (dim_indices / channels))
    # Compute the rotation angle for every (position, dimension pair)
    # Multiplying them creates a table
    angles = positions[:, None] * inv_freq[None, :]

    # Computes cosine and sine angle for every rotation angle
    cos = torch.cos(angles).repeat_interleave(2, dim=-1)
    sin = torch.sin(angles).repeat_interleave(2, dim=-1)
    # Add batch dimension, allowing PyTorch to automatically broadcast same position
    # information across every example in the batch
    cos = cos.unsqueeze(0)
    sin = sin.unsqueeze(0)

    return (x * cos) + (rotate_half(x) * sin) # Rotary embedding
