"""Metrics for evaluating continual learning methods."""

from typing import List

import numpy as np


def average_accuracy(results_matrix: np.ndarray) -> float:
    """Calculate the average accuracy across all tasks.
    
    This computes the mean of all diagonal elements in the results matrix,
    representing the average performance on each task after training on itself.
    
    Args:
        results_matrix: A 2D numpy array where results_matrix[i][j] represents
            the accuracy on task j after training on task i.
            
    Returns:
        The average accuracy as a float.
    """
    num_tasks = results_matrix.shape[0]
    diag = np.diag(results_matrix)
    return float(np.mean(diag))


def backward_transfer(results_matrix: np.ndarray) -> float:
    """Calculate backward transfer between tasks.
    
    Backward transfer measures how much learning a new task hurts
    performance on previously learned tasks. Negative values indicate
    interference (forgetting).
    
    Args:
        results_matrix: A 2D numpy array where results_matrix[i][j] represents
            the accuracy on task j after training on task i.
            
    Returns:
        The backward transfer value. Negative values indicate forgetting.
    """
    num_tasks = results_matrix.shape[0]
    off_diagonal = []
    for i in range(num_tasks):
        for j in range(i):
            off_diagonal.append(results_matrix[i][j] - results_matrix[i - 1][j])
    return float(np.mean(off_diagonal)) if off_diagonal else 0.0


def forward_transfer(results_matrix: np.ndarray) -> float:
    """Calculate forward transfer between tasks.
    
    Forward transfer measures how much learning a new task helps
    performance on future tasks. Positive values indicate beneficial
    transfer of knowledge.
    
    Args:
        results_matrix: A 2D numpy array where results_matrix[i][j] represents
            the accuracy on task j after training on task i.
            
    Returns:
        The forward transfer value. Positive values indicate beneficial transfer.
    """
    num_tasks = results_matrix.shape[0]
    forward_scores = []
    for i in range(num_tasks):
        for j in range(i + 1, num_tasks):
            forward_scores.append(results_matrix[i][j])
    return float(np.mean(forward_scores)) if forward_scores else 0.0


def forgetting_measure(results_matrix: np.ndarray) -> float:
    """Calculate the forgetting measure.
    
    Forgetting measures the drop in performance on each task after
    subsequent tasks have been learned. It computes the maximum drop
    in accuracy for each task compared to its performance right after
    training on that task.
    
    Args:
        results_matrix: A 2D numpy array where results_matrix[i][j] represents
            the accuracy on task j after training on task i.
            
    Returns:
        The forgetting measure as a float. Higher values indicate more forgetting.
    """
    num_tasks = results_matrix.shape[0]
    forgetting = []
    for j in range(num_tasks):
        best_after_training = results_matrix[j][j]
        drops = []
        for i in range(j + 1, num_tasks):
            drop = best_after_training - results_matrix[i][j]
            drops.append(drop)
        if drops:
            forgetting.append(max(drops))
    return float(np.mean(forgetting)) if forgetting else 0.0
