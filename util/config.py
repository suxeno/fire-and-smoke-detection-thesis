"""
Configuration utilities for loading and validating YAML configs.
Used by main.py to load training configurations.
"""
import yaml
import argparse
from pathlib import Path


def load_config(config_path):
    """
    Load YAML configuration file.
    
    Args:
        config_path: Path to YAML config file
    
    Returns:
        argparse.Namespace with config parameters
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Convert dict to namespace for easy attribute access
    args = argparse.Namespace(**config_dict)
    return args


def validate_config(args):
    """
    Validate configuration parameters and setup directories.
    
    Args:
        args: argparse.Namespace with config
    
    Returns:
        Validated args
    """
    # Check data path exists
    data_path = Path(args.data_path)
    if not data_path.exists():
        raise ValueError(f"Data path does not exist: {data_path}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check CUDA availability
    if args.device == 'cuda':
        import torch
        if not torch.cuda.is_available():
            print("⚠ WARNING: CUDA not available, falling back to CPU")
            args.device = 'cpu'
    
    return args
