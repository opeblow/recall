"""Experience Replay method for continual learning."""

from typing import Any, Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from research.methods.base import BaseMethod


class ReplayMethod(BaseMethod):
    """Experience Replay method for continual learning.
    
    This method stores a small subset of examples from previous tasks
    and replays them during training on new tasks to prevent forgetting.
    """
    
    def __init__(self, model: nn.Module, config: Dict[str, Any]) -> None:
        """Initialize the replay method.
        
        Args:
            model: The neural network model.
            config: Configuration dictionary.
        """
        super().__init__(model, config)
        self.lr = config.get("lr", 0.001)
        self.epochs = config.get("epochs", 10)
        self.memory_size = config.get("memory_size", 100)
        self.replay_buffer: List[torch.Tensor] = []
        self.replay_labels: List[torch.Tensor] = []
    
    def train_task(self, task_id: int, train_loader: DataLoader) -> Dict[str, Any]:
        """Train on a specific task with experience replay.
        
        Args:
            task_id: The ID of the current task.
            train_loader: DataLoader for the current task's training data.
            
        Returns:
            Dictionary containing training metrics.
        """
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()
        
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        for epoch in range(self.epochs):
            for batch in train_loader:
                inputs, targets = batch
                
                if self.replay_buffer:
                    replay_data = torch.cat(self.replay_buffer)
                    replay_labels = torch.cat(self.replay_labels)
                    replay_inputs = replay_data[:inputs.size(0)]
                    replay_targets = replay_labels[:inputs.size(0)]
                    
                    combined_inputs = torch.cat([inputs, replay_inputs])
                    combined_targets = torch.cat([targets, replay_targets])
                else:
                    combined_inputs = inputs
                    combined_targets = targets
                
                optimizer.zero_grad()
                outputs = self.model(combined_inputs)
                loss = criterion(outputs, combined_targets)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total_correct += (predicted[:inputs.size(0)] == targets).sum().item()
                total_samples += targets.size(0)
        
        self._update_memory(train_loader)
        
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        avg_accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        return {"loss": avg_loss, "accuracy": avg_accuracy, "epoch": self.epochs}
    
    def _update_memory(self, train_loader: DataLoader) -> None:
        """Update the replay memory with samples from current task.
        
        Args:
            train_loader: DataLoader for the current task's training data.
        """
        samples_per_task = self.memory_size
        
        self.model.eval()
        for batch in train_loader:
            inputs, targets = batch
            if len(self.replay_buffer) * inputs.size(0) < self.memory_size:
                self.replay_buffer.append(inputs)
                self.replay_labels.append(targets)
            else:
                break
        
        if len(self.replay_buffer) > samples_per_task:
            self.replay_buffer = self.replay_buffer[:samples_per_task]
            self.replay_labels = self.replay_labels[:samples_per_task]
    
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
        
        with torch.no_grad():
            for batch in test_loader:
                inputs, targets = batch
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        accuracy = correct / total if total > 0 else 0.0
        return {"accuracy": accuracy}
