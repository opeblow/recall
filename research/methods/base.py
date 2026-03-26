"""Abstract base class for continual learning methods."""

from abc import ABC, abstractmethod
from typing import Any, Dict

import torch.nn as nn


class BaseMethod(ABC):
    """Abstract base class for continual learning methods.
    
    This class defines the interface that all continual learning
    methods must implement.
    
    Attributes:
        model: The neural network model.
        config: Configuration dictionary for the method.
        memory: Dictionary to store any method-specific data.
    """
    
    def __init__(self, model: nn.Module, config: Dict[str, Any], logger: "Logger" = None) -> None:
        """Initialize the method with model and configuration.
        
        Args:
            model: The neural network model to train.
            config: Configuration dictionary containing method-specific settings.
        """
        self.model = model
        self.config = config
        self.memory: Dict[str, Any] = {}
    
    @abstractmethod
    def train_task(self, task_id: int, train_loader: Any) -> Dict[str, Any]:
        """Train on a specific task.
        
        Args:
            task_id: The ID of the current task.
            train_loader: DataLoader for the current task's training data.
            
        Returns:
            Dictionary containing training metrics.
        """
        pass
    
    @abstractmethod
    def evaluate(self, task_id: int, test_loader: Any) -> Dict[str, float]:
        """Evaluate on a specific task.
        
        Args:
            task_id: The ID of the task to evaluate on.
            test_loader: DataLoader for the current task's test data.
            
        Returns:
            Dictionary containing evaluation metrics.
        """
        pass
    
    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        """Log metrics for tracking.
        
        Args:
            metrics: Dictionary of metric names and values.
            step: The current step or iteration number.
        """
        pass
