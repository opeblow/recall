"""Elastic Weight Consolidation (EWC) method for continual learning."""

from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from research.methods.base import BaseMethod
from research.utils.logger import Logger


class EWCMethod(BaseMethod):
    """Elastic Weight Consolidation (EWC) method.
    
    EWC adds a regularization term to the loss that penalizes changes
    to important parameters (as measured by Fisher information).
    """
    
    def __init__(self, model: nn.Module, config: Dict[str, Any], logger: Logger = None) -> None:
        """Initialize the EWC method.
        
        Args:
            model: The neural network model.
            config: Configuration dictionary.
            logger: Logger for wandb logging.
        """
        super().__init__(model, config)
        self.logger = logger
        self.lr = config.get("lr", 0.001)
        self.epochs = config.get("epochs", 10)
        self.ewc_lambda = config.get("ewc_lambda", 5000)
        self.memory["fisher_dict"] = {}
        self.memory["optpar_dict"] = {}
        self.memory["task_count"] = 0
    
    def train_task(self, task_id: int, train_loader: DataLoader) -> Dict[str, Any]:
        """Train on a specific task with EWC regularization.
        
        Args:
            task_id: The ID of the current task.
            train_loader: DataLoader for the current task's training data.
            
        Returns:
            Dictionary containing training metrics.
        """
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()
        ewc_lambda = self.ewc_lambda
        epochs = self.epochs
        
        if task_id > 0:
            self._compute_fisher(self.model, train_loader)
        
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
                
                if task_id > 0:
                    ewc_loss = self._ewc_loss()
                    loss += ewc_lambda * ewc_loss
                
                ewc_loss_val = 0.0
                if task_id > 0:
                    ewc_loss_val = self._ewc_loss().item()
                    loss += ewc_lambda * ewc_loss_val
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                epoch_correct += (predicted == targets).sum().item()
                epoch_samples += targets.size(0)
                
                pbar.set_postfix({"loss": loss.item(), "ewc_loss": ewc_lambda * ewc_loss_val})
            
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
                    "ewc_loss": ewc_lambda * self._ewc_loss().item() if task_id > 0 else 0,
                    "phase": "train"
                }, step=task_id * epochs + epoch)
        
        self._save_optpar()
        self.memory["task_count"] += 1
        
        avg_loss = total_loss / epochs
        avg_accuracy = total_correct / total_samples
        return {"loss": avg_loss, "accuracy": avg_accuracy, "epoch": epochs}
    
    def _compute_fisher(self, model: nn.Module, train_loader: DataLoader) -> None:
        """Compute Fisher information matrix for parameter importance.
        
        Args:
            model: The neural network model.
            train_loader: DataLoader for computing Fisher information.
        """
        model.eval()
        fisher_dict = {}
        
        for name, param in model.named_parameters():
            fisher_dict[name] = torch.zeros_like(param.data)
        
        num_samples = 0
        for batch in train_loader:
            inputs, targets = batch
            outputs = model(inputs)
            log_probs = torch.log_softmax(outputs, dim=1)
            loss = nn.functional.cross_entropy(outputs, targets)
            model.zero_grad()
            loss.backward()
            
            for name, param in model.named_parameters():
                if param.grad is not None:
                    fisher_dict[name] += param.grad.data ** 2
            
            num_samples += inputs.size(0)
        
        for name in fisher_dict:
            if num_samples > 0:
                fisher_dict[name] /= num_samples
        
        self.memory["fisher_dict"] = fisher_dict
    
    def _ewc_loss(self) -> torch.Tensor:
        """Compute EWC regularization loss."""
        loss = 0.0
        fisher_dict = self.memory.get("fisher_dict", {})
        optpar_dict = self.memory.get("optpar_dict", {})
        
        for name, param in self.model.named_parameters():
            if name in optpar_dict and name in fisher_dict:
                loss += (fisher_dict[name] * (param - optpar_dict[name]) ** 2).sum()
        
        return loss
    
    def _save_optpar(self) -> None:
        """Save optimal parameters after training a task."""
        optpar_dict = {}
        for name, param in self.model.named_parameters():
            optpar_dict[name] = param.data.clone()
        self.memory["optpar_dict"] = optpar_dict
    
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