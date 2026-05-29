# PyTorch GPT From Scratch (Transformer)

This project implements a GPT-style decoder-only Transformer from scratch in PyTorch.

The goal is to understand how Transformers work internally.

The main project focuses on:
- tiktoken tokenization
- masked self-attention
- multi-head attention
- feed-forward networks
- residual connections
- layer normalization
- stacked Transformer blocks
- next-token prediction
- baseline training
- torch.compile and mixed precision optimization
- Distributed Data Parallel training

Later extensions:
- original Transformer encoder
- encoder-decoder cross-attention
- larger scale training
- Rotary Position Embeddings