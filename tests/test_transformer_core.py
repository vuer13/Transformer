from pathlib import Path

import pytest
import torch

from transformer.attention import MultiHeadAttention, SelfAttentionHead
from transformer.block import FeedForward, LayerNorm, TransformerBlock
from transformer.config import Config
from transformer.data import TextDatasetConfig, TextTokenDataset
from transformer.model import Model

def small_config() -> Config:
    return Config(
        vocab_size=100,
        block_size=8,
        n_embd=32,
        n_head=4,
        n_layer=2,
        dropout=0.0,
        bias=True,
    )


def test_config_head_size() -> None:
    config = small_config()

    assert config.head_size == 8


def test_config_rejects_invalid_head_size() -> None:
    with pytest.raises(ValueError):
        Config(
            vocab_size=100,
            block_size=8,
            n_embd=30,
            n_head=8,
            n_layer=2,
            dropout=0.0,
            bias=True,
        )


def test_text_dataset_get_batch_shapes(tmp_path: Path) -> None:
    input_path = tmp_path / "input.txt"
    input_path.write_text(
        "hello world this is a tiny transformer dataset " * 100,
        encoding="utf-8",
    )

    dataset = TextTokenDataset(
        TextDatasetConfig(
            input_path=str(input_path),
            block_size=8,
            encoding_name="gpt2",
            device="cpu",
        )
    )

    x, y = dataset.get_batch(split="train", batch_size=4)

    assert x.shape == (4, 8)
    assert y.shape == (4, 8)
    assert x.dtype == torch.long
    assert y.dtype == torch.long


def test_self_attention_head_shape() -> None:
    config = small_config()
    head = SelfAttentionHead(config)

    x = torch.randn(2, 8, 32)
    out = head(x)

    assert out.shape == (2, 8, 8)


def test_multi_head_attention_shape() -> None:
    config = small_config()
    attention = MultiHeadAttention(config)

    x = torch.randn(2, 8, 32)
    out = attention(x)

    assert out.shape == (2, 8, 32)


def test_feed_forward_shape() -> None:
    config = small_config()
    feed_forward = FeedForward(config)

    x = torch.randn(2, 8, 32)
    out = feed_forward(x)

    assert out.shape == (2, 8, 32)


def test_layer_norm_shape() -> None:
    config = small_config()
    layer_norm = LayerNorm(config)

    x = torch.randn(2, 8, 32)
    out = layer_norm(x)

    assert out.shape == (2, 8, 32)


def test_transformer_block_shape() -> None:
    config = small_config()
    block = TransformerBlock(config)

    x = torch.randn(2, 8, 32)
    out = block(x)

    assert out.shape == (2, 8, 32)


def test_gpt_model_logits_and_loss_shape() -> None:
    config = small_config()
    model = Model(config)

    idx = torch.randint(0, config.vocab_size, (2, 8))
    targets = torch.randint(0, config.vocab_size, (2, 8))

    logits, loss = model(idx, targets)

    assert logits.shape == (2, 8, config.vocab_size)
    assert loss is not None
    assert loss.shape == torch.Size([])


def test_generate_returns_expected_length() -> None:
    config = small_config()
    model = Model(config)
    model.eval()

    start = torch.zeros((1, 1), dtype=torch.long)
    generated = model.generate(start, max_new_tokens=5)

    assert generated.shape == (1, 6)


def test_gpt_model_with_rope_shape() -> None:
    config = Config(
        vocab_size=100,
        block_size=8,
        n_embd=32,
        n_head=4,
        n_layer=2,
        dropout=0.0,
        bias=True,
        use_rope=True,
    )

    model = Model(config)

    idx = torch.randint(0, config.vocab_size, (2, 8))
    targets = torch.randint(0, config.vocab_size, (2, 8))

    logits, loss = model(idx, targets)

    assert logits.shape == (2, 8, config.vocab_size)
    assert loss is not None
    assert loss.shape == torch.Size([])
