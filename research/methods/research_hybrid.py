"""ResearchHybrid method combining EWC, selective replay, and dynamic task heads."""

from typing import Any, Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from research.methods.base import BaseMethod
from research.utils.logger import Logger


class ResearchHybridMethod(BaseMethod):
    """ResearchHybrid method combining EWC, selective replay, and dynamic task heads.
    
    This novel method combines:
    - EWC regularization on backbone weights only
    - Selective replay using highest softmax entropy samples (hardest examples)
    - Dynamic task heads (one shared backbone + separate output head per task)
    """
    
    def __init__(self, model: nn.Module, config: Dict[str, Any], logger: Logger = None) -> None:
        """Initialize the ResearchHybrid method.
        
        Args:
            model: The neural network model (backbone).
            config: Configuration dictionary.
            logger: Logger for wandb logging.
        """
        super().__init__(model, config)
        self.logger = logger
        self.lr = config.get("lr", 0.001)
        self.epochs = config.get("epochs", 10)
        self.ewc_lambda = config.get("ewc_lambda", 1000)
        self.replay_size = config.get("replay_size", 300)
        self.heads_lr = config.get("heads_lr", 0.01)
        self.input_dim = config.get("input_dim", 784)
        
        self.memory["fisher_dict"] = {}
        self.memory["optpar_dict"] = {}
        self.memory["task_heads"] = {}
        self.memory["task_count"] = 0
        self.memory["replay_data"] = []
        self.memory["replay_labels"] = []
        self.memory["replay_task_ids"] = []
        
        self.criterion = nn.CrossEntropyLoss()
    
    def train_task(self, task_id: int, train_loader: DataLoader) -> Dict[str, Any]:
        """Train on a specific task with hybrid approach.
        
        Args:
            task_id: The ID of the current task.
            train_loader: DataLoader for the current task's training data.
            
        Returns:
            Dictionary containing training metrics.
        """
        num_classes = self._get_num_classes(train_loader)
        self._create_task_head(task_id, num_classes)
        
        head_params = [self.memory["task_heads"][task_id]]
        backbone_params = list(self.model.parameters())
        
        optimizer = torch.optim.Adam([
            {"params": backbone_params, "lr": self.lr},
            {"params": head_params, "lr": self.heads_lr}
        ])
        
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
                
                replay_inputs, replay_targets, replay_task_ids = self._get_replay_samples()
                
                if replay_inputs is not None:
                    combined_inputs = torch.cat([inputs, replay_inputs])
                    combined_targets = torch.cat([targets, replay_targets])
                    combined_task_ids = torch.cat([torch.full((inputs.size(0),), task_id, dtype=torch.long), replay_task_ids])
                else:
                    combined_inputs = inputs
                    combined_targets = targets
                    combined_task_ids = torch.full((inputs.size(0),), task_id, dtype=torch.long)
                
                optimizer.zero_grad()
                
                features = self.model(combined_inputs)
                
                outputs_list = []
                for tid in combined_task_ids.unique():
                    head = self.memory["task_heads"][tid.item()]
                    mask = combined_task_ids == tid
                    task_features = features[mask]
                    task_outputs = head(task_features)
                    outputs_list.append((task_outputs, mask))
                
                outputs = torch.zeros_like(combined_targets, dtype=torch.float)
                for task_outputs, mask in outputs_list:
                    outputs[mask] = task_outputs.squeeze()
                
                loss = self.criterion(outputs.unsqueeze(1), combined_targets)
                
                if task_id > 0:
                    ewc_loss = self._ewc_loss()
                    loss += self.ewc_lambda * ewc_loss
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item() * inputs.size(0)
                with torch.no_grad():
                    final_features = self.model(inputs)
                    final_head = self.memory["task_heads"][task_id]
                    final_outputs = final_head(final_features)
                    _, predicted = torch.max(final_outputs, 1)
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
        
        self._save_optpar()
        self._selective_replay(train_loader, task_id)
        self.memory["task_count"] += 1
        
        avg_loss = total_loss / epochs
        avg_accuracy = total_correct / total_samples
        return {"loss": avg_loss, "accuracy": avg_accuracy, "epoch": epochs}
    
    def _get_num_classes(self, train_loader: DataLoader) -> int:
        """Get number of classes from data.
        
        Args:
            train_loader: DataLoader for training data.
            
        Returns:
            Number of classes.
        """
        for batch in train_loader:
            _, targets = batch
            return int(targets.max().item() + 1)
        return 10
    
    def _create_task_head(self, task_id: int, num_classes: int) -> None:
        """Create a new output head for the task.
        
        Args:
            task_id: The ID of the task.
            num_classes: Number of classes for this task.
        """
        self.model.eval()
        with torch.no_grad():
            dummy_input = torch.randn(1, self.input_dim)
            try:
                dummy_output = self.model(dummy_input)
                feat_dim = dummy_output.shape[1]
            except:
                feat_dim = 128
        
        head = nn.Linear(feat_dim, 1).to(next(self.model.parameters()).device)
        self.memory["task_heads"][task_id] = head
    
    def _compute_fisher(self, model: nn.Module, train_loader: DataLoader) -> None:
        """Compute Fisher information matrix for EWC on backbone only.
        
        Args:
            model: The backbone model.
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
        """Compute EWC regularization loss on backbone only."""
        loss = 0.0
        fisher_dict = self.memory.get("fisher_dict", {})
        optpar_dict = self.memory.get("optpar_dict", {})
        
        backbone_param_names = set()
        for name, _ in self.model.named_parameters():
            backbone_param_names.add(name)
        
        for name, param in self.model.named_parameters():
            if name in backbone_param_names and name in optpar_dict and name in fisher_dict:
                loss += (fisher_dict[name] * (param - optpar_dict[name]) ** 2).sum()
        
        return loss
    
    def _save_optpar(self) -> None:
        """Save optimal parameters after training a task."""
        optpar_dict = {}
        for name, param in self.model.named_parameters():
            optpar_dict[name] = param.data.clone()
        self.memory["optpar_dict"] = optpar_dict
    
    def _selective_replay(self, train_loader: DataLoader, task_id: int) -> None:
        """Select the k most uncertain samples (highest softmax entropy) per task.
        
        Args:
            train_loader: DataLoader for the current task's training data.
            task_id: The ID of the current task.
        """
        self.model.eval()
        
        all_inputs = []
        all_targets = []
        all_entropies = []
        
        with torch.no_grad():
            for batch in train_loader:
                inputs, targets = batch
                outputs = self.model(inputs)
                probs = torch.softmax(outputs, dim=1)
                entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=1)
                
                all_inputs.append(inputs)
                all_targets.append(targets)
                all_entropies.append(entropy)
        
        all_inputs = torch.cat(all_inputs)
        all_targets = torch.cat(all_targets)
        all_entropies = torch.cat(all_entropies)
        
        _, top_indices = torch.topk(all_entropies, min(self.replay_size, len(all_entropies)))
        
        self.memory["replay_data"].append(all_inputs[top_indices])
        self.memory["replay_labels"].append(all_targets[top_indices])
        self.memory["replay_task_ids"].append(torch.full((len(top_indices),), task_id, dtype=torch.long))
    
    def _get_replay_samples(self) -> tuple:
        """Get replay samples from memory.
        
        Returns:
            Tuple of (replay_inputs, replay_labels, replay_task_ids).
        """
        if not self.memory["replay_data"]:
            return None, None, None
        
        all_inputs = torch.cat(self.memory["replay_data"])
        all_labels = torch.cat(self.memory["replay_labels"])
        all_task_ids = torch.cat(self.memory["replay_task_ids"])
        
        return all_inputs, all_labels, all_task_ids
    
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
        
        head = self.memory["task_heads"].get(task_id)
        if head is None:
            return {"accuracy": 0.0, "loss": float('inf')}
        
        with torch.no_grad():
            for batch in test_loader:
                inputs, targets = batch
                features = self.model(inputs)
                outputs = head(features)
                loss = self.criterion(outputs.squeeze(), targets)
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