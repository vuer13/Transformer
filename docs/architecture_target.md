# Architecture Target

This project implements a GPT-style decoder-only Transformer from scratch in PyTorch.

## Main Model

- [ ] token embeddings
- [ ] positional embeddings
- [x] masked self-attention
- [x] multi-head self-attention
- [x] feed-forward network
- [ ] residual connections
- [ ] layer normalization
- [ ] stacked Transformer blocks
- [ ] final language modeling head
- [ ] next-token prediction loss
- [ ] autoregressive text generation

## Second Stage

The following are not part of the first implementation:

- encoder stack
- encoder-decoder cross-attention
- translation-style sequence-to-sequence training
- beam search
- Rotary Position Embeddings
- Distributed Data Parallel
- torch.compile
- mixed precision

These will be added only after the core GPT model works.

## Optional Extension

After the GPT-style model is complete, this project may add the original Transformer encoder and encoder-decoder cross-attention for comparison.