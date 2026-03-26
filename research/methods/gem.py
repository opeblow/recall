"""Gradient Episodic Memory (GEM) method for continual learning."""

from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from research.methods.base import BaseMethod
from research.utils.logger import Logger


class GEMMethod(BaseMethod):
    """Gradient Episodic Memory (GEM) method.
    
    GEM stores a small buffer of examples from previous tasks and
    constrains gradients to prevent interference with previous tasks.
    """
    
    def __init__(self, model: nn.Module, config: Dict[str, Any], logger: Logger = None) -> None:
        """Initialize the GEM method.
        
        Args:
            model: The neural network model.
            config: Configuration dictionary.
            logger: Logger for wandb logging.
        """
        super().__init__(model, config)
        self.logger = logger
        self.lr = config.get("lr", 0.001)
        self.epochs = config.get("epochs", 10)
        self.memory_size = config.get("memory_size", 200)
        self.memory[" episodic_data"] = []
        self.memory["episodic_labels"] = []
    
    def train_task(self, task_id: int, train_loader: DataLoader) -> Dict[str, Any]:
        """Train on a specific task with GEM regularization.
        
        Args:
            task_id: The ID of the current task.
            train_loader: DataLoader for the current task's training data.
            
        Returns:
            Dictionary containing training metrics.
        """
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()
        epochs = self.epochs
        
        self._collect_memory(task_id, train_loader)
        
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
                
                if task_id > 0 and self.memory["episodic_data"]:
                    grad = self._compute_gradient(inputs, targets, criterion)
                    
                    ref_grads = self._compute_reference_gradients(criterion)
                    
                    if ref_grads is not None:
                        grad = self._project_gradient(grad, ref_grads)
                    
                    self._apply_gradient(optimizer, grad)
                else:
                    optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                
                with torch.no_grad():
                    outputs = self.model(inputs)
                    loss = criterion(outputs, targets)
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
    
    def _collect_memory(self, task_id: int, train_loader: DataLoader) -> None:
        """Collect samples from current task for episodic memory.
        
        Args:
            task_id: The ID of the current task.
            train_loader: DataLoader for the current task's training data.
        """
        self.model.eval()
        
        all_inputs = []
        all_targets = []
        
        for batch in train_loader:
            inputs, targets = batch
            all_inputs.append(inputs)
            all_targets.append(targets)
        
        all_inputs = torch.cat(all_inputs)
        all_targets = torch.cat(all_targets)
        
        perm = torch.randperm(all_inputs.size(0))
        indices = perm[:self.memory_size]
        
        self.memory["episodic_data"].append(all_inputs[indices])
        self.memory["episodic_labels"].append(all_targets[indices])
    
    def _compute_gradient(self, inputs: torch.Tensor, targets: torch.Tensor, 
                         criterion: nn.Module) -> List[torch.Tensor]:
        """Compute gradient for current task.
        
        Args:
            inputs: Input tensors.
            targets: Target tensors.
            criterion: Loss function.
            
        Returns:
            List of gradients for each parameter.
        """
        self.model.zero_grad()
        outputs = self.model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        
        grad = []
        for param in self.model.parameters():
            if param.grad is not None:
                grad.append(param.grad.data.clone())
            else:
                grad.append(None)
        return grad
    
    def _compute_reference_gradients(self, criterion: nn.Module) -> List[List[torch.Tensor]]:
        """Compute gradients on episodic memory for reference.
        
        Args:
            criterion: Loss function.
            
        Returns:
            List of reference gradients for each parameter.
        """
        if not self.memory["episodic_data"]:
            return None
        
        ref_grads = []
        
        for task_data, task_labels in zip(self.memory["episodic_data"], self.memory["episodic_labels"]):
            self.model.zero_grad()
            outputs = self.model(task_data)
            loss = criterion(outputs, task_labels)
            loss.backward()
            
            task_grads = []
            for param in self.model.parameters():
                if param.grad is not None:
                    task_grads.append(param.grad.data.clone())
                else:
                    task_grads.append(None)
            ref_grads.append(task_grads)
        
        return ref_grads
    
    def _project_gradient(self, grad: List[torch.Tensor], 
                          ref_grads: List[List[torch.Tensor]]) -> List[torch.Tensor]:
        """Project gradient to nearest gradient that doesn't increase past task loss.
        
        Uses a closed-form approximation for quadratic programming.
        
        Args:
            grad: Current task gradients.
            ref_grads: List of gradients for each past task.
            
        Returns:
            Projected gradients.
        """
        if ref_grads is None or len(ref_grads) == 0:
            return grad
        
        ref_grads_array = []
        for task_grads in ref_grads:
            task_grad_flat = []
            for g in task_grads:
                if g is not None:
                    task_grad_flat.append(g.flatten())
            if task_grad_flat:
                ref_grads_array.append(torch.cat(task_grad_flat))
        
        if not ref_grads_array:
            return grad
        
        grad_flat = []
        for g in grad:
            if g is not None:
                grad_flat.append(g.flatten())
        grad_flat = torch.cat(grad_flat)
        
        ref_matrix = torch.stack(ref_grads_array)
        
        dots = torch.matmul(ref_matrix, grad_flat)
        
        if torch.all(dots >= 0):
            return grad
        
        constraints = ref_matrix
        constraint_targets = torch.zeros(len(ref_grads_array))
        
        try:
            Q = torch.eye(grad_flat.size(0))
            Q_pinv = torch.pinverse(Q)
            
            mask = dots < 0
            if mask.sum() > 0:
                active_constraints = constraints[mask]
                active_targets = constraint_targets[mask]
                
                if active_constraints.size(0) > 0 and active_constraints.size(1) > 0:
                    A = torch.matmul(active_constraints, active_constraints.T)
                    b = torch.matmul(active_constraints, grad_flat)
                    
                    try:
                        lagrange_mult = torch.linalg.solve(A + 1e-6 * torch.eye(A.size(0)), b)
                        lagrange_mult = torch.clamp(lagrange_mult, min=0)
                        
                        correction = torch.matmul(active_constraints.T, lagrange_mult)
                        grad_flat = grad_flat - correction
                    except:
                        pass
        except:
            pass
        
        idx = 0
        projected_grad = []
        for g in grad:
            if g is not None:
                numel = g.numel()
                projected_grad.append(grad_flat[idx:idx+numel].view_as(g))
                idx += numel
            else:
                projected_grad.append(None)
        
        return projected_grad
    
    def _apply_gradient(self, optimizer: torch.optim.Optimizer, 
                       grad: List[torch.Tensor]) -> None:
        """Apply projected gradient to model.
        
        Args:
            optimizer: Optimizer for gradient application.
            grad: Projected gradients.
        """
        optimizer.zero_grad()
        
        for param, g in zip(self.model.parameters(), grad):
            if param.grad is not None and g is not None:
                param.grad.data = g
        
        optimizer.step()
    
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