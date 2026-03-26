"""Naive baseline method for continual learning."""

from typing import Any, Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from research.methods.base import BaseMethod
from research.utils.logger import Logger
from research.utils.config import Config


class NaiveMethod(BaseMethod):
    """Naive baseline method that trains on each task sequentially.
    
    This method simply trains the model on each task without any
    continual learning technique. It serves as a baseline to compare
    against other methods.
    """
    
    def __init__(self, model: nn.Module, config: Dict[str, Any], logger: Logger = None) -> None:
        """Initialize the naive method.
        
        Args:
            model: The neural network model.
            config: Configuration dictionary.
            logger: Logger for wandb logging.
        """
        super().__init__(model, config)
        self.logger = logger
        self.lr = config.get("lr", 0.001)
        self.epochs = config.get("epochs", 10)
    
    def train_task(self, task_id: int, train_loader: DataLoader) -> Dict[str, Any]:
        """Train on a specific task without any regularization.
        
        Args:
            task_id: The ID of the current task.
            train_loader: DataLoader for the current task's training data.
            
        Returns:
            Dictionary containing training metrics.
        """
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()
        epochs = self.epochs
        
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_samples = 0
            
            pbar = tqdm(train_loader, desc=f"Task {task_id} Epoch {epoch+1}/{epochs}")
            for batch in pbar:
                inputs, targets = batch
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                epoch_correct += (predicted == targets).sum().item()
                epoch_samples += targets.size(0)
                
                pbar.set_postfix({"loss": loss.item()})
            
            avg_epoch_loss = epoch_loss / epoch_samples
            epoch_accuracy = epoch_correct / epoch_samples
            total_loss += avg_epoch_loss
            total_correct += epoch_correct
            total_samples += epoch_samples
            
            if self.logger:
                self.logger.log({
                    "task_id": task_id,
                    "epoch": epoch,
                    "loss": avg_epoch_loss,
                    "accuracy": epoch_accuracy,
                    "phase": "train"
                }, step=task_id * epochs + epoch)
        
        avg_loss = total_loss / epochs
        avg_accuracy = total_correct / total_samples
        return {"loss": avg_loss, "accuracy": avg_accuracy, "epoch": epochs}
    
    def evaluate(self, task_id: int, test_loader: DataLoader) -> Dict[str, float]:
        """Evaluate on a specific task.
        
        Args:
            task_id: The ID of the task to evaluate on.
            test_loader: DataLoader for the current task's test data.
            
        Returns:
            Dictionary containing evaluation metrics.
        """
        self.model.eval()
        correct = 0
        total = 0
        total_loss = 0.0
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for batch in test_loader:
                inputs, targets = batch
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                total_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        accuracy = correct / total if total > 0 else 0.0
        avg_loss = total_loss / total if total > 0 else 0.0
        
        if self.logger:
            self.logger.log({
                "task_id": task_id,
                "loss": avg_loss,
                "accuracy": accuracy,
                "phase": "eval"
            }, step=task_id)
        
        return {"accuracy": accuracy, "loss": avg_loss}