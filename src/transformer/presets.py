from transformer.config import Config


def get_local_debug_config(vocab_size: int) -> Config:
    """
    Small config for local CPU/MPS development.

    This is meant for quick tests and debugging.
    """
    return Config(
        vocab_size=vocab_size,
        block_size=32,
        n_embd=64,
        n_head=4,
        n_layer=2,
        dropout=0.1,
        bias=True,
        use_rope=False,
    )


def get_80m_config(vocab_size: int = 50257) -> Config:
    """
    Approximate 80M parameter GPT-style decoder model.

    This config is intended for CUDA/GPU experiments, not local CPU/MPS training.

    RoPE is enabled, so learned position embeddings are skipped.
    """
    return Config(
        vocab_size=vocab_size,
        block_size=512,
        n_embd=512,
        n_head=8,
        n_layer=9,
        dropout=0.1,
        bias=True,
        use_rope=True,
    )