# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Model registry for fire and smoke detection.

Three model variants for ablation study:
1. DETR (detr.py) - Standard DETR with CNN backbone, no superpixels
2. DETR-SLIC (detr_slic.py) - Superpixel-only, no CNN backbone  
3. DETR-Hybrid (detr_hybrid.py) - CNN backbone + superpixel features

Usage:
    # In config yaml:
    model_name: 'detr'        # Standard DETR
    model_name: 'detr_slic'   # Superpixel-only (lightweight)
    model_name: 'detr_hybrid' # CNN + Superpixels
"""

from .detr import build
from .detr_slic import build_detr_slic
from .detr_hybrid import build_detr_hybrid


def build_model(args):
    """
    Build detection model based on config.
    
    Args:
        args: Config namespace with 'model_name' or 'model_type' attribute
        
    Returns:
        model, criterion, postprocessors
    """
    # Use model_type if available (original model name before subset suffix)
    # Fall back to model_name for backward compatibility
    model_name = getattr(args, 'model_type', None) or getattr(args, 'model_name', 'detr')
    model_name = model_name.lower()
    
    # DETR-SLIC: Superpixel-only, no CNN backbone
    if model_name == 'detr_slic':
        return build_detr_slic(args)
    
    # DETR-Hybrid: CNN backbone + superpixel features
    if model_name == 'detr_hybrid':
        return build_detr_hybrid(args)
    
    # Standard DETR: CNN backbone only
    return build(args)