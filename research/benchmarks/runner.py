"""Benchmark runner for continual learning experiments."""

from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from research.benchmarks.metrics import (
    average_accuracy,
    backward_transfer,
    forgetting_measure,
    forward_transfer,
)


class BenchmarkRunner:
    """Runner for executing continual learning benchmarks.
    
    This class manages the execution of continual learning experiments,
    including training on multiple tasks and evaluating performance.
    
    Attributes:
        model: The neural network model.
        method: The continual learning method to use.
        device: The device to run computations on.
    """
    
    def __init__(
        self,
        model: nn.Module,
        method: Any,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ) -> None:
        """Initialize the benchmark runner.
        
        Args:
            model: The neural network model.
            method: The continual learning method instance.
            device: The device to run computations on.
        """
        self.model = model.to(device)
        self.method = method
        self.device = device
        self.results_matrix: List[List[float]] = []
    
    def run(
        self,
        train_loaders: List[DataLoader],
        test_loaders: List[DataLoader]
    ) -> Dict[str, float]:
        """Run the benchmark across all tasks.
        
        Args:
            train_loaders: List of DataLoaders for each task's training data.
            test_loaders: List of DataLoaders for each task's test data.
            
        Returns:
            Dictionary containing final benchmark metrics.
        """
        num_tasks = len(train_loaders)
        
        for task_id in range(num_tasks):
            self.method.train_task(task_id, train_loaders[task_id])
            
            task_results = []
            for eval_task_id in range(num_tasks):
                metrics = self.method.evaluate(eval_task_id, test_loaders[eval_task_id])
                task_results.append(metrics["accuracy"])
            
            self.results_matrix.append(task_results)
        
        return self.get_metrics()
    
    def get_metrics(self) -> Dict[str, float]:
        """Compute final benchmark metrics from results matrix.
        
        Returns:
            Dictionary containing all benchmark metrics.
        """
        results = np.array(self.results_matrix)
        
        metrics = {
            "average_accuracy": average_accuracy(results),
            "backward_transfer": backward_transfer(results),
            "forward_transfer": forward_transfer(results),
            "forgetting_measure": forgetting_measure(results),
        }
        
        return metrics
