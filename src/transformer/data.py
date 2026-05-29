from dataclasses import dataclass
from pathlib import Path

import torch
import tiktoken
from jaxtyping import Int
from torch import Tensor


@dataclass(frozen=True)
class TextDatasetConfig:
    """Congifuration for loadaing and tokenizing a text dataset."""
    input_path: str
    block_size: int         # Number of tokens in each input sequence
    encoding: str = "gpt2"  # Tokenizer encoding to use

    train_split: float = 0.9  # Proportion of data to use for training
    device: str = "cpu"       # Device to load the dataset on


class TextTokenDataset:
    """
    Load raw text data, tokenize it, and create input-target pairs for training a transformer model.
    
    x = input tokens
    y = target tokens (shifted by one position)
    """

    def __init__(self, config: TextDatasetConfig):
        self.config = config
        self.block_size = config.block_size
        self.device = config.device

        if not 0.0 < config.train_split < 1.0:
            raise ValueError("train_split must be between 0 and 1")
        
        # Read the raw text data
        text = Path(config.input_path).read_text(encoding="utf-8")
        # Load the tokenizer
        self.encoder = tiktoken.get_encoding(config.encoding)
        # Convert text to token IDs
        token_ids = self.encoder.encode(text)
        # Store token IDs as PyTorch tensor
        data = torch.tensor(token_ids, dtype=torch.long)
        # Split data into training and validation sets
        split_idx = int(len(data) * config.train_split)
        self.train_data = data[:split_idx].to(self.device)
        self.val_data = data[split_idx:].to(self.device)
        # Number of unique tokens in the vocabulary
        self.vocab_size = self.encoder.n_vocab

    def encode(self, text: str) -> list[int]:
        """Convert text into token IDs"""
        return self.encoder.encode(text)
    
    def decode(self, token_ids: list[int]) -> str:
        """Convert token IDs back into text"""
        return self.encoder.decode(token_ids)
    
    def get_batch(
        self, 
        split: str,
        batch_size: int   
    ) -> tuple[
        Int[Tensor, "batch time"], 
        Int[Tensor, "batch time"]
    ]:
        """
       Randomly sample batch of token sequences

       Returns:
            x: (batch_size, block_size) input token IDs
            y: (batch_size, block_size) target token IDs (shifted by one position)
        """
        if split == "train":
            data = self.train_data
        elif split == "val":
            data = self.val_data
        else:
            raise ValueError("split must be either 'train' or 'val'")

        if len(data) <= self.block_size:
            raise ValueError(
                f"Not enough tokens in {split} split. "
                f"Need more than block_size={self.block_size} tokens."
            )

        # Pick random starting positions.
        starts = torch.randint(
            low=0,
            high=len(data) - self.block_size,
            size=(batch_size,),
        )

        # Input sequence.
        x = torch.stack([
            data[start : start + self.block_size]
            for start in starts
        ])
        # Target sequence: same window shifted one token forward.
        y = torch.stack([
            data[start + 1 : start + self.block_size + 1]
            for start in starts
        ])

        return x.to(self.device), y.to(self.device)