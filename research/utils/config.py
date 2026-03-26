"""Configuration loader for research project."""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


class Config:
    """Configuration manager for loading and accessing config values.
    
    This class loads configuration from a YAML file and provides
    easy access to configuration values with defaults.
    
    Attributes:
        config: The loaded configuration dictionary.
    """
    
    def __init__(self, config_path: Optional[str] = None) -> None:
        """Initialize the configuration from a YAML file.
        
        Args:
            config_path: Path to the config file. Defaults to config.yaml in project root.
        """
        if config_path is None:
            config_path = os.path.join(Path(__file__).parent.parent, "config.yaml")
        
        self.config = self._load_config(config_path)
    
    def _load_config(self, path: str) -> Dict[str, Any]:
        """Load configuration from YAML file.
        
        Args:
            path: Path to the config file.
            
        Returns:
            Configuration dictionary.
        """
        config_path = Path(path)
        if not config_path.exists():
            return {}
        
        with open(config_path, 'r') as f:
            return yaml.safe_load(f) or {}
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value by key.
        
        Supports nested keys using dot notation (e.g., "training.lr").
        
        Args:
            key: The configuration key (supports dot notation for nested values).
            default: Default value if key is not found.
            
        Returns:
            The configuration value or default.
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_method_config(self, method_name: str) -> Dict[str, Any]:
        """Get configuration for a specific method.
        
        Args:
            method_name: Name of the method (e.g., "ewc", "replay", "gem").
            
        Returns:
            Configuration dictionary for the method.
        """
        method_key = method_name.lower()
        return self.config.get(method_key, {})
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration.
        
        Returns:
            Model configuration dictionary.
        """
        return self.config.get("model", {})
    
    def get_training_config(self) -> Dict[str, Any]:
        """Get training configuration.
        
        Returns:
            Training configuration dictionary.
        """
        return self.config.get("training", {})
    
    def get_benchmark_config(self) -> Dict[str, Any]:
        """Get benchmark configuration.
        
        Returns:
            Benchmark configuration dictionary.
        """
        return self.config.get("benchmark", {})
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration.
        
        Returns:
            Logging configuration dictionary.
        """
        return self.config.get("logging", {})
    
    def get_device_config(self) -> Dict[str, Any]:
        """Get device configuration.
        
        Returns:
            Device configuration dictionary.
        """
        return self.config.get("device", {})


def load_config(config_path: Optional[str] = None) -> Config:
    """Load configuration from a YAML file.
    
    Args:
        config_path: Optional path to config file.
        
    Returns:
        Config object.
    """
    return Config(config_path)