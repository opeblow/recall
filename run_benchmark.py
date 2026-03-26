"""Run benchmark script for continual learning experiments."""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from research.benchmarks.runner import BenchmarkRunner
from research.methods.ewc import EWCMethod
from research.methods.gem import GEMMethod
from research.methods.naive import NaiveMethod
from research.methods.replay import ReplayMethod
from research.methods.research_hybrid import ResearchHybridMethod as HybridMethod
from research.utils.logger import Logger
from research.utils.config import load_config


def create_dummy_data(num_samples: int, num_classes: int, input_dim: int) -> DataLoader:
    """Create dummy data for testing.
    
    Args:
        num_samples: Number of samples per task.
        num_classes: Number of classes per task.
        input_dim: Input dimension.
        
    Returns:
        DataLoader with dummy data.
    """
    data = torch.randn(num_samples, input_dim)
    labels = torch.randint(0, num_classes, (num_samples,))
    dataset = TensorDataset(data, labels)
    return DataLoader(dataset, batch_size=32, shuffle=True)


def get_method(name: str, model: nn.Module, config: dict) -> nn.Module:
    """Get a continual learning method by name.
    
    Args:
        name: Name of the method.
        model: The neural network model.
        config: Configuration dictionary.
        
    Returns:
        An instance of the requested method.
    """
    methods = {
        "naive": NaiveMethod,
        "ewc": EWCMethod,
        "replay": ReplayMethod,
        "gem": GEMMethod,
        "hybrid": HybridMethod,
    }
    
    if name not in methods:
        raise ValueError(f"Unknown method: {name}")
    
    return methods[name](model, config)


def main() -> None:
    """Main function to run benchmarks."""
    parser = argparse.ArgumentParser(description="Run continual learning benchmarks")
    parser.add_argument("--method", type=str, default="naive", help="Method to use")
    parser.add_argument("--num_tasks", type=int, default=None, help="Number of tasks")
    parser.add_argument("--epochs", type=int, default=None, help="Epochs per task")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--config", type=str, default=None, help="Path to config JSON")
    parser.add_argument("--yaml_config", type=str, default=None, help="Path to config YAML")
    
    args = parser.parse_args()
    
    cfg = load_config(args.yaml_config)
    
    config = {
        "epochs": args.epochs if args.epochs is not None else cfg.get("training.epochs", 10),
        "lr": args.lr if args.lr is not None else cfg.get("training.lr", 0.001),
        "batch_size": cfg.get("training.batch_size", 32),
        "weight_decay": cfg.get("training.weight_decay", 0.0001),
    }
    
    num_tasks = args.num_tasks if args.num_tasks is not None else cfg.get("benchmark.num_tasks", 5)
    input_dim = cfg.get("model.input_dim", 784)
    hidden_dim = cfg.get("model.hidden_dim", 256)
    output_dim = cfg.get("model.output_dim", 10)
    samples_per_task = cfg.get("benchmark.samples_per_task", 100)
    test_samples = cfg.get("benchmark.test_samples_per_task", 50)
    
    method_config = cfg.get_method_config(args.method)
    for key, value in method_config.items():
        if key not in config:
            config[key] = value
    
    if args.config and Path(args.config).exists():
        with open(args.config) as f:
            config.update(json.load(f))
    
    logger_config = cfg.get_logging_config()
    logger = Logger(project_name=logger_config.get("project_name", "research"))
    
    model = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim)
    )
    
    method = get_method(args.method, model, config)
    
    train_loaders = [
        create_dummy_data(samples_per_task, output_dim, input_dim)
        for _ in range(num_tasks)
    ]
    test_loaders = [
        create_dummy_data(test_samples, output_dim, input_dim)
        for _ in range(num_tasks)
    ]
    
    runner = BenchmarkRunner(model, method)
    results = runner.run(train_loaders, test_loaders)
    
    logger.log(results, step=num_tasks)
    
    print(f"Results for method: {args.method}")
    for key, value in results.items():
        print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    main()
