"""Logging utilities for research project."""

import os
from pathlib import Path
from typing import Any, Dict

import dotenv
import torch
import wandb


class Logger:
    """Logger class for experiment tracking and model checkpoints.
    
    This class handles logging metrics to Weights & Biases and
    saving model checkpoints.
    
    Attributes:
        project_name: The name of the wandb project.
        run: The wandb run object.
    """
    
    def __init__(self, project_name: str = "research") -> None:
        """Initialize the logger with wandb.
        
        Args:
            project_name: The name of the wandb project. Defaults to "research".
        """
        dotenv.load_dotenv()
        api_key = os.getenv("WAND_API_KEY")
        if api_key:
            wandb.login(key=api_key)
        
        self.project_name = project_name
        self.run = wandb.init(project=project_name)
    
    def log(self, metrics: Dict[str, Any], step: int) -> None:
        """Log metrics to wandb.
        
        Args:
            metrics: Dictionary of metric names and values.
            step: The current step or iteration number.
        """
        self.run.log(metrics, step=step)
    
    def save_checkpoint(self, model: torch.nn.Module, path: str) -> None:
        """Save a model checkpoint.
        
        Args:
            model: The PyTorch model to save.
            path: The file path to save the checkpoint.
        """
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), path)
        wandb.save(path)
