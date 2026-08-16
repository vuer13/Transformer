import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim import Optimizer

from transformer.config import Config


def count_parameters(model: nn.Module) -> int:
    """Counts how many trainable and non-trainable parameters are inside the model."""

    return sum(parameter.numel() for parameter in model.parameters())


def save_config(config: Config, output_path: Path) -> None:
    """Saves the model config to a JSON file."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(asdict(config), file, indent=2)


def save_checkpoint(
    model: nn.Module,
    optimizer: Optimizer,
    step: int,
    config: Config,
    output_path: Path,
    losses: dict[str, float] | None = None,
) -> None:
    """Saves training progress to a checkpoint file"""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "step": step,
        "config": asdict(config),
        "model_state_dict": model.state_dict(), # model weights saved
        "optimizer_state_dict": optimizer.state_dict(), # optimizer state
        "losses": losses or {},
    }

    torch.save(checkpoint, output_path)


def load_checkpoint(
    model: nn.Module,
    optimizer: Optimizer,
    checkpoint_path: Path,
    device: str,
) -> int:
    """Loads model and optimizer progress from a checkpoint file."""

    checkpoint: dict[str, Any] = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"]) # restore model state
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"]) # restore optimizer state
    step = int(checkpoint["step"]) # last completed step

    return step + 1


def append_loss_row(
    output_path: Path,
    step: int,
    train_loss: float,
    val_loss: float,
) -> None:
    """Adds one row of train/validation loss to a CSV file."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    file_exists = output_path.exists()

    with output_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["step", "train_loss", "val_loss"],
        )

        # Only write the header once
        if not file_exists:
            writer.writeheader()

        writer.writerow(
            {
                "step": step,
                "train_loss": train_loss,
                "val_loss": val_loss,
            }
        )
