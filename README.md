# PyTorch GPT From Scratch (Transformer)

This project implements a GPT-style decoder-only Transformer from scratch in PyTorch.

The goal is to understand how Transformers work internally by building each major component step by step.

## Core Features

- `tiktoken` tokenization
- training data pipeline
- bigram baseline model
- masked self-attention
- multi-head attention
- feed-forward networks
- residual connections
- layer normalization
- stacked Transformer blocks
- next-token prediction loss
- autoregressive text generation
- baseline training on Tiny Shakespeare
- checkpoint saving and loading
- loss logging
- pytest shape and generation tests

## Performance and Scaling Features

- optional CUDA mixed precision training
- optional `torch.compile` support
- Distributed Data Parallel training script
- Rotary Position Embeddings
- approximate 80M parameter model preset for larger GPU experiments

## Later Extensions

- architecture diagrams
- original Transformer encoder
- encoder-decoder cross-attention
- larger-scale training experiments
- improved sampling strategies
- more advanced datasets