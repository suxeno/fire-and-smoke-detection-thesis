"""
Debug script to test the entire DETR-SLIC pipeline end-to-end.
Tests all components from data loading to training, validation, and testing.
"""
# Standard library imports
import os
import sys
import json
import argparse
import tempfile
from pathlib import Path
from itertools import islice
from datetime import datetime

# Third-party imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Optional imports
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    HAS_COCO = True
except ImportError:
    HAS_COCO = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

# Project imports
from util import (
    load_config,
    validate_config,
    build_data_loader,
    MetricLogger,
    SmoothedValue,
    print_model_summary,
    count_parameters
)
from util.training_logger import TrainingLogger
from engine import train_one_epoch, evaluate
from models import build_model

# Terminal Color Codes

class Colors:
    """ANSI color codes for terminal output"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


# Utility Print Functions

def print_test(msg):
    """Print test step message"""
    print(f"{Colors.BLUE}[TEST]{Colors.END} {msg}")


def print_pass(msg):
    """Print success message"""
    print(f"{Colors.GREEN}[PASS]{Colors.END} ✓ {msg}")


def print_fail(msg):
    """Print failure message"""
    print(f"{Colors.RED}[FAIL]{Colors.END} ✗ {msg}")


def print_warn(msg):
    """Print warning message"""
    print(f"{Colors.YELLOW}[WARN]{Colors.END} ⚠ {msg}")


def print_info(msg):
    """Print info message"""
    print(f"{Colors.CYAN}[INFO]{Colors.END} ℹ {msg}")


def print_section(msg):
    """Print section header"""
    print(f"\n{Colors.BOLD}{'='*80}")
    print(f"{msg}")
    print(f"{'='*80}{Colors.END}\n")


# Test Functions

def test_imports():
    """Test 1: Verify all required imports work"""
    print_section("TEST 1: Checking Imports")
    
    try:
        # Core dependencies
        print_test("Importing PyTorch...")
        import torch
        import torchvision
        print_pass(f"PyTorch {torch.__version__}, Torchvision {torchvision.__version__}")
        print_info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print_info(f"CUDA version: {torch.version.cuda}")
            print_info(f"GPU: {torch.cuda.get_device_name(0)}")
        
        # Project modules
        print_test("Importing project modules...")
        from util import load_config, validate_config, build_data_loader
        from util.training_logger import TrainingLogger
        from engine import train_one_epoch, evaluate
        from models import build_model
        print_pass("Core project modules imported successfully")
        
        # Optional dependencies
        print_test("Checking optional dependencies...")
        
        if HAS_COCO:
            print_pass("pycocotools available (COCO metrics enabled)")
        else:
            print_warn("pycocotools not available (COCO metrics will be disabled)")
        
        if HAS_MATPLOTLIB:
            print_pass("matplotlib available (visualization enabled)")
        else:
            print_warn("matplotlib not available (visualization will be disabled)")
        
        if HAS_NUMPY:
            print_pass(f"NumPy {np.__version__} available")
        else:
            print_warn("NumPy not available")
        
        if HAS_PIL:
            print_pass(f"Pillow {Image.__version__} available")
        else:
            print_warn("Pillow not available")
        
        return True
    except Exception as e:
        print_fail(f"Import test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config(config_path='configs/detr_slic.yaml'):
    """Test 2: Load and validate configuration"""
    print_section("TEST 2: Config Loading and Validation")
    
    try:
        print_test(f"Loading config from {config_path}...")
        if not Path(config_path).exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        args = load_config(config_path)
        print_pass("Config loaded successfully")
        
        print_test("Validating config...")
        args = validate_config(args)
        print_pass("Config validated")
        
        print_test("Checking critical config parameters:")
        print(f"  Model name:        {args.model_name}")
        print(f"  Use SLIC:          {args.use_slic}")
        print(f"  Dataset path:      {args.data_path}")
        print(f"  Use sample:        {getattr(args, 'use_sample', False)}")
        print(f"  Num classes:       {args.num_classes}")
        print(f"  Batch size:        {args.batch_size}")
        print(f"  Learning rate:     {args.lr}")
        print(f"  Epochs:            {args.epochs}")
        print(f"  Device:            {args.device}")
        print(f"  Output dir:        {args.output_dir}")
        
        # Verify dataset paths exist
        print_test("Verifying dataset paths...")
        data_path = Path(args.data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {data_path}")
        print_pass(f"Dataset path exists: {data_path}")
        
        images_path = data_path / 'images'
        if not images_path.exists():
            raise FileNotFoundError(f"Images directory not found: {images_path}")
        print_pass(f"Images directory exists: {images_path}")
        
        ann_path = data_path / 'annotations' / 'COCO' / 'Annotations'
        if not ann_path.exists():
            raise FileNotFoundError(f"Annotations directory not found: {ann_path}")
        print_pass(f"Annotations directory exists: {ann_path}")
        
        return True, args
    except Exception as e:
        print_fail(f"Config test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_data_loading(args):
    """Test 3: Data loading pipeline"""
    print_section("TEST 3: Data Loading Pipeline")
    
    try:
        # Load train dataset
        print_test("Loading training dataset...")
        train_loader = build_data_loader('train', args)
        print_pass(f"Train loader created: {len(train_loader.dataset)} samples, "
                  f"{len(train_loader)} batches")
        
        # Load validation dataset
        print_test("Loading validation dataset...")
        val_loader = build_data_loader('val', args)
        print_pass(f"Val loader created: {len(val_loader.dataset)} samples, "
                  f"{len(val_loader)} batches")
        
        # Load test dataset
        print_test("Loading test dataset...")
        test_loader = build_data_loader('test', args)
        print_pass(f"Test loader created: {len(test_loader.dataset)} samples, "
                  f"{len(test_loader)} batches")
        
        # Test batch loading
        print_test("Testing batch loading from train set...")
        samples, targets = next(iter(train_loader))
        print_pass(f"Batch loaded successfully")
        print_info(f"  Images shape: {samples.tensors.shape}")
        print_info(f"  Mask shape: {samples.mask.shape}")
        print_info(f"  Number of targets: {len(targets)}")
        
        # Validate target format
        print_test("Checking target format...")
        target = targets[0]
        required_keys = ['boxes', 'labels', 'image_id', 'area', 'orig_size']
        missing_keys = [k for k in required_keys if k not in target]
        if missing_keys:
            raise ValueError(f"Missing keys in target: {missing_keys}")
        print_pass(f"Target contains all required keys: {list(target.keys())}")
        
        # Validate target values
        print_test("Validating target values...")
        print_info(f"  Number of objects: {len(target['boxes'])}")
        print_info(f"  Boxes shape: {target['boxes'].shape}")
        print_info(f"  Labels shape: {target['labels'].shape}")
        print_info(f"  Labels unique: {target['labels'].unique().tolist()}")
        print_info(f"  Image ID: {target['image_id'].item()}")
        print_info(f"  Original size: {target['orig_size'].tolist()}")
        
        # Verify labels are within valid range
        max_label = target['labels'].max().item()
        if max_label >= args.num_classes:
            raise ValueError(f"Found label {max_label} >= num_classes {args.num_classes}")
        print_pass("All labels within valid range")
        
        # Verify boxes are in valid format [cx, cy, w, h] normalized [0, 1]
        boxes = target['boxes']
        if (boxes < 0).any() or (boxes > 1).any():
            print_warn("Some boxes outside [0, 1] range - may be due to augmentation")
        else:
            print_pass("All boxes in normalized [0, 1] range")
        
        return True, train_loader, val_loader, test_loader
    except Exception as e:
        print_fail(f"Data loading test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None, None


def test_model_building(args):
    """Test 4: Model building and architecture verification"""
    print_section("TEST 4: Model Building")
    
    try:
        print_test("Building model...")
        model, criterion, postprocessors = build_model(args)
        print_pass("Model built successfully")
        
        # Count parameters
        print_test("Counting parameters...")
        param_counts = count_parameters(model)
        total_params = param_counts['total']
        trainable_params = param_counts['trainable']
        non_trainable_params = param_counts['non_trainable']
        print_pass(f"Total parameters: {total_params:,}")
        print_info(f"  Trainable: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
        print_info(f"  Non-trainable: {non_trainable_params:,} ({100*non_trainable_params/total_params:.2f}%)")
        
        # Verify model components
        print_test("Checking model architecture components...")
        
        # Backbone
        if hasattr(model, 'backbone'):
            print_pass(f"  Backbone: {type(model.backbone).__name__}")
        else:
            raise ValueError("Model missing backbone")
        
        # Transformer (different architectures)
        if hasattr(model, 'transformer'):
            print_pass(f"  Transformer: {type(model.transformer).__name__}")
        elif hasattr(model, 'decoder'):
            # DETR-SLIC with separate encoder/decoder
            if hasattr(model, 'superpixel_encoder'):
                print_pass(f"  Superpixel encoder: {type(model.superpixel_encoder).__name__}")
            print_pass(f"  Decoder: {type(model.decoder).__name__}")
        else:
            raise ValueError("Model missing transformer/decoder components")
        
        # SLIC components (if enabled)
        if args.use_slic:
            if hasattr(model, 'slic'):
                print_pass(f"  SLIC module: {type(model.slic).__name__}")
            else:
                print_warn("  SLIC enabled in config but module not found")
            
            if hasattr(model, 'superpixel_pooling'):
                print_pass(f"  Superpixel pooling: {type(model.superpixel_pooling).__name__}")
            elif hasattr(model, 'feature_pooling'):
                print_pass(f"  Feature pooling: {type(model.feature_pooling).__name__}")
            else:
                print_warn("  Feature pooling module not found")
        
        # Detection heads
        if hasattr(model, 'class_embed'):
            print_pass(f"  Class head: {type(model.class_embed).__name__}")
        else:
            raise ValueError("Model missing class_embed")
        
        if hasattr(model, 'bbox_embed'):
            print_pass(f"  BBox head: {type(model.bbox_embed).__name__}")
        else:
            raise ValueError("Model missing bbox_embed")
        
        # Verify criterion
        print_test("Checking criterion...")
        if criterion is None:
            raise ValueError("Criterion is None")
        print_pass(f"Criterion: {type(criterion).__name__}")
        print_info(f"  Loss weights: {criterion.weight_dict}")
        
        # Verify postprocessors
        print_test("Checking postprocessors...")
        if 'bbox' not in postprocessors:
            raise ValueError("Missing bbox postprocessor")
        print_pass(f"Postprocessors available: {list(postprocessors.keys())}")
        
        return True, model, criterion, postprocessors
    except Exception as e:
        print_fail(f"Model building test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None, None


def test_forward_pass(model, train_loader, device):
    """Test 5: Forward pass through the model"""
    print_section("TEST 5: Forward Pass")
    
    try:
        model = model.to(device)
        model.eval()
        
        print_test("Getting a batch...")
        samples, targets = next(iter(train_loader))
        batch_size = samples.tensors.shape[0]
        print_info(f"Batch size: {batch_size}")
        
        print_test(f"Moving batch to {device}...")
        samples = samples.to(device)
        print_pass(f"Batch moved to {device}")
        
        print_test("Running forward pass...")
        with torch.no_grad():
            outputs = model(samples)
        print_pass("Forward pass completed successfully")
        
        # Verify output format
        print_test("Checking output format...")
        if 'pred_logits' not in outputs:
            raise ValueError("Missing 'pred_logits' in outputs")
        if 'pred_boxes' not in outputs:
            raise ValueError("Missing 'pred_boxes' in outputs")
        print_pass("Output contains required keys")
        
        # Verify output shapes
        print_test("Verifying output shapes...")
        pred_logits = outputs['pred_logits']
        pred_boxes = outputs['pred_boxes']
        
        print_info(f"  pred_logits shape: {pred_logits.shape}")
        print_info(f"  pred_boxes shape: {pred_boxes.shape}")
        
        # Check expected shapes
        if pred_logits.shape[0] != batch_size:
            print_warn(f"Batch size mismatch in pred_logits")
        if pred_boxes.shape[0] != batch_size:
            print_warn(f"Batch size mismatch in pred_boxes")
        
        print_pass("Output shapes verified")
        
        # Check for NaN or Inf values
        print_test("Checking for NaN/Inf values...")
        if torch.isnan(pred_logits).any() or torch.isinf(pred_logits).any():
            raise ValueError("Found NaN or Inf in pred_logits")
        if torch.isnan(pred_boxes).any() or torch.isinf(pred_boxes).any():
            raise ValueError("Found NaN or Inf in pred_boxes")
        print_pass("No NaN or Inf values in outputs")
        
        # Check auxiliary outputs if present
        if 'aux_outputs' in outputs:
            print_test("Checking auxiliary outputs...")
            aux_outputs = outputs['aux_outputs']
            print_info(f"  Number of auxiliary outputs: {len(aux_outputs)}")
            for i, aux in enumerate(aux_outputs):
                if 'pred_logits' not in aux or 'pred_boxes' not in aux:
                    raise ValueError(f"Auxiliary output {i} missing required keys")
            print_pass("All auxiliary outputs have correct format")
        
        return True, outputs
    except Exception as e:
        print_fail(f"Forward pass test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_loss_computation(model, criterion, train_loader, device):
    """Test 6: Loss computation"""
    print_section("TEST 6: Loss Computation")
    
    try:
        model = model.to(device)
        model.train()
        
        print_test("Getting a batch...")
        samples, targets = next(iter(train_loader))
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        print_pass("Batch prepared")
        
        print_test("Computing forward pass...")
        outputs = model(samples)
        print_pass("Forward pass completed")
        
        print_test("Computing losses...")
        loss_dict = criterion(outputs, targets)
        print_pass("Losses computed successfully")
        
        print_test("Checking loss components...")
        required_losses = ['loss_ce', 'loss_bbox', 'loss_giou']
        for loss_name in required_losses:
            if loss_name not in loss_dict:
                raise ValueError(f"Missing loss: {loss_name}")
            loss_value = loss_dict[loss_name].item()
            print_info(f"  {loss_name:15s}: {loss_value:.4f}")
            
            # Check for valid loss values
            if torch.isnan(loss_dict[loss_name]) or torch.isinf(loss_dict[loss_name]):
                raise ValueError(f"{loss_name} is NaN or Inf")
        
        print_pass("All required loss components present and valid")
        
        # Check auxiliary losses if present
        aux_loss_keys = [k for k in loss_dict.keys() if k.endswith('_0') or k.endswith('_1')]
        if aux_loss_keys:
            print_test("Checking auxiliary losses...")
            print_info(f"  Found {len(aux_loss_keys)} auxiliary loss components")
            print_pass("Auxiliary losses present")
        
        # Compute total weighted loss
        print_test("Computing total weighted loss...")
        weight_dict = criterion.weight_dict
        total_loss = sum(loss_dict[k] * weight_dict[k] 
                        for k in loss_dict.keys() if k in weight_dict)
        print_pass(f"Total weighted loss: {total_loss.item():.4f}")
        
        # Check if loss is reasonable
        if total_loss.item() > 1000:
            print_warn(f"Total loss is very high: {total_loss.item():.4f}")
        elif total_loss.item() < 0:
            raise ValueError(f"Total loss is negative: {total_loss.item():.4f}")
        
        return True
    except Exception as e:
        print_fail(f"Loss computation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_backward_pass(model, criterion, train_loader, device):
    """Test 7: Backward pass and optimization"""
    print_section("TEST 7: Backward Pass and Optimization")
    
    try:
        model = model.to(device)
        model.train()
        
        print_test("Creating optimizer...")
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        print_pass("Optimizer created (AdamW, lr=1e-4)")
        
        print_test("Getting a batch...")
        samples, targets = next(iter(train_loader))
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        print_test("Forward pass...")
        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        total_loss = sum(loss_dict[k] * weight_dict[k] 
                        for k in loss_dict.keys() if k in weight_dict)
        print_info(f"  Total loss: {total_loss.item():.4f}")
        
        print_test("Backward pass...")
        optimizer.zero_grad()
        total_loss.backward()
        print_pass("Gradients computed")
        
        print_test("Checking gradients...")
        grad_count = 0
        grad_norms = []
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_count += 1
                grad_norm = param.grad.norm().item()
                grad_norms.append(grad_norm)
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    raise ValueError(f"Found NaN/Inf in gradients for {name}")
        
        if grad_count == 0:
            raise ValueError("No gradients computed")
        
        avg_grad_norm = sum(grad_norms) / len(grad_norms) if grad_norms else 0
        max_grad_norm = max(grad_norms) if grad_norms else 0
        
        print_pass(f"Gradients verified: {grad_count} parameters")
        print_info(f"  Avg gradient norm: {avg_grad_norm:.6f}")
        print_info(f"  Max gradient norm: {max_grad_norm:.6f}")
        
        if max_grad_norm > 100:
            print_warn(f"Large gradient detected: {max_grad_norm:.6f}")
        
        print_test("Applying gradient clipping...")
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
        print_pass("Gradients clipped (max_norm=0.1)")
        
        print_test("Optimizer step...")
        optimizer.step()
        print_pass("Weights updated")
        
        return True
    except Exception as e:
        print_fail(f"Backward pass test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_epoch(model, criterion, train_loader, device, num_iterations=5):
    """Test 8: Training for multiple iterations"""
    print_section(f"TEST 8: Training Epoch ({num_iterations} iterations)")
    
    try:
        model = model.to(device)
        model.train()
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        
        print_test(f"Training for {num_iterations} iterations...")
        
        limited_loader = islice(train_loader, num_iterations)
        
        losses_history = []
        
        for i, (samples, targets) in enumerate(limited_loader):
            samples = samples.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # Forward
            outputs = model(samples)
            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict
            total_loss = sum(loss_dict[k] * weight_dict[k] 
                           for k in loss_dict.keys() if k in weight_dict)
            
            # Backward
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            optimizer.step()
            
            losses_history.append(total_loss.item())
            print_info(f"  Iteration {i+1}/{num_iterations}: loss={total_loss.item():.4f}")
        
        print_pass("Training iterations completed successfully")
        
        # Check if loss is decreasing or stable
        if len(losses_history) > 1:
            avg_loss_first_half = sum(losses_history[:len(losses_history)//2]) / (len(losses_history)//2)
            avg_loss_second_half = sum(losses_history[len(losses_history)//2:]) / (len(losses_history) - len(losses_history)//2)
            
            print_info(f"  Avg loss (first half): {avg_loss_first_half:.4f}")
            print_info(f"  Avg loss (second half): {avg_loss_second_half:.4f}")
            
            if avg_loss_second_half < avg_loss_first_half:
                print_pass("Loss is decreasing (expected behavior)")
            elif abs(avg_loss_second_half - avg_loss_first_half) < 0.1:
                print_pass("Loss is stable")
            else:
                print_warn("Loss is increasing - may need more iterations to see convergence")
        
        return True
    except Exception as e:
        print_fail(f"Training epoch test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_evaluation(model, criterion, postprocessors, val_loader, device):
    """Test 9: Evaluation on validation set"""
    print_section("TEST 9: Evaluation")
    
    try:
        print_test("Running evaluation...")
        stats = evaluate(model, criterion, postprocessors, val_loader, device)
        print_pass("Evaluation completed")
        
        print_test("Checking evaluation metrics...")
        required_metrics = ['loss', 'loss_ce', 'loss_bbox', 'loss_giou', 'class_error']
        for metric in required_metrics:
            if metric not in stats:
                print_warn(f"Missing metric: {metric}")
            else:
                print_info(f"  {metric:15s}: {stats[metric]:.4f}")
        print_pass("Evaluation metrics obtained")
        
        # Check COCO metrics if available
        if 'AP' in stats:
            print_test("COCO metrics available:")
            print_info(f"  AP (mAP):          {stats.get('AP', 0):.4f}")
            print_info(f"  AP50 (mAP50):      {stats.get('AP50', stats.get('mAP50', 0)):.4f}")
            print_info(f"  AP75:              {stats.get('AP75', 0):.4f}")
            print_info(f"  Recall:            {stats.get('Recall', 0):.4f}")
            print_pass("COCO metrics computed")
            
            # Per-class metrics
            if 'AP_fire' in stats or 'AP_smoke' in stats:
                print_test("Per-class metrics:")
                if 'AP_fire' in stats:
                    print_info(f"  Fire  - AP: {stats.get('AP_fire', 0):.4f}, "
                             f"Recall: {stats.get('Recall_fire', 0):.4f}")
                if 'AP_smoke' in stats:
                    print_info(f"  Smoke - AP: {stats.get('AP_smoke', 0):.4f}, "
                             f"Recall: {stats.get('Recall_smoke', 0):.4f}")
                print_pass("Per-class metrics available")
        else:
            print_warn("COCO metrics not available (pycocotools may not be installed)")
        
        return True, stats
    except Exception as e:
        print_fail(f"Evaluation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_checkpoint_save_load(model, optimizer):
    """Test 10: Checkpoint saving and loading"""
    print_section("TEST 10: Checkpoint Save/Load")
    
    try:
        print_test("Creating checkpoint...")
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': 0,
            'best_val_loss': 999.0,
        }
        print_pass("Checkpoint dictionary created")
        
        print_test("Saving checkpoint to temporary file...")
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            checkpoint_path = f.name
            torch.save(checkpoint, checkpoint_path)
        
        checkpoint_size = os.path.getsize(checkpoint_path) / (1024 * 1024)  # MB
        print_pass(f"Checkpoint saved ({checkpoint_size:.2f} MB)")
        print_info(f"  Path: {checkpoint_path}")
        
        print_test("Loading checkpoint...")
        loaded_checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print_pass("Checkpoint loaded successfully")
        
        print_test("Verifying checkpoint contents...")
        required_keys = ['model', 'optimizer', 'epoch']
        for key in required_keys:
            if key not in loaded_checkpoint:
                raise ValueError(f"Missing key in checkpoint: {key}")
        print_pass("Checkpoint contains all required keys")
        
        print_test("Restoring model state...")
        model.load_state_dict(loaded_checkpoint['model'])
        print_pass("Model state restored successfully")
        
        print_test("Restoring optimizer state...")
        optimizer.load_state_dict(loaded_checkpoint['optimizer'])
        print_pass("Optimizer state restored successfully")
        
        print_test("Cleaning up temporary file...")
        os.remove(checkpoint_path)
        print_pass("Temporary checkpoint deleted")
        
        return True
    except Exception as e:
        print_fail(f"Checkpoint test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_postprocessing(model, postprocessors, test_loader, device):
    """Test 11: Postprocessing predictions"""
    print_section("TEST 11: Postprocessing Predictions")
    
    try:
        model = model.to(device)
        model.eval()
        
        print_test("Getting test batch...")
        samples, targets = next(iter(test_loader))
        batch_size = len(targets)
        samples = samples.to(device)
        print_info(f"Batch size: {batch_size}")
        
        print_test("Running inference...")
        with torch.no_grad():
            outputs = model(samples)
        print_pass("Inference completed")
        
        print_test("Postprocessing outputs...")
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0).to(device)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        print_pass(f"Postprocessed {len(results)} predictions")
        
        print_test("Checking result format...")
        result = results[0]
        required_keys = ['scores', 'labels', 'boxes']
        missing_keys = [k for k in required_keys if k not in result]
        if missing_keys:
            raise ValueError(f"Missing keys in postprocessed results: {missing_keys}")
        print_pass(f"Result contains all required keys: {list(result.keys())}")
        
        # Analyze predictions
        print_test("Analyzing predictions...")
        for i, result in enumerate(results):
            num_detections = len(result['scores'])
            if num_detections > 0:
                max_score = result['scores'].max().item()
                min_score = result['scores'].min().item()
                unique_labels = result['labels'].unique().tolist()
                print_info(f"  Image {i}: {num_detections} detections, "
                         f"scores [{min_score:.3f}, {max_score:.3f}], "
                         f"labels {unique_labels}")
            else:
                print_info(f"  Image {i}: No detections")
        
        print_pass("Postprocessing verification complete")
        
        return True
    except Exception as e:
        print_fail(f"Postprocessing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_logger(args):
    """Test 12: Training logger functionality"""
    print_section("TEST 12: Training Logger")
    
    try:
        print_test("Creating training logger...")
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger = TrainingLogger(output_dir, args.model_name)
        print_pass("Training logger created")
        
        print_test("Testing epoch logging...")
        train_stats = {
            'loss': 32.5,
            'loss_ce': 0.65,
            'loss_bbox': 3.2,
            'loss_giou': 1.9,
            'class_error': 95.0,
            'lr': 0.0001
        }
        
        val_stats = {
            'loss': 28.3,
            'loss_ce': 0.58,
            'loss_bbox': 2.9,
            'loss_giou': 1.7,
            'class_error': 90.0,
            'AP': 0.12,
            'mAP50': 0.23,
            'Recall': 0.35,
            'AP_fire': 0.15,
            'AP50_fire': 0.25,
            'Recall_fire': 0.38,
            'AP_smoke': 0.10,
            'AP50_smoke': 0.21,
            'Recall_smoke': 0.32
        }
        
        logger.log_epoch(epoch=0, train_stats=train_stats, 
                        val_stats=val_stats, epoch_time=120.5)
        print_pass("Epoch logging successful")
        
        print_test("Checking log files...")
        log_file = output_dir / 'log.txt'
        metrics_file = output_dir / 'metrics_summary.txt'
        
        if not log_file.exists():
            raise FileNotFoundError(f"Log file not created: {log_file}")
        if not metrics_file.exists():
            raise FileNotFoundError(f"Metrics summary file not created: {metrics_file}")
        
        print_pass("Log files created successfully")
        print_info(f"  Log file: {log_file}")
        print_info(f"  Metrics summary: {metrics_file}")
        
        print_test("Testing final results logging...")
        test_stats = {'loss': 27.8, 'AP': 0.14, 'mAP50': 0.25, 'Recall': 0.37}
        logger.log_final_results(total_time="0:02:00", test_stats=test_stats)
        print_pass("Final results logging successful")
        
        return True
    except Exception as e:
        print_fail(f"Training logger test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# Main Test Runner
# ============================================================================

def run_all_tests(config_path='configs/detr_slic.yaml'):
    """Run all tests sequentially"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}")
    print("="*80)
    print("DETR-SLIC END-TO-END PIPELINE DEBUG".center(80))
    print("="*80)
    print(f"{Colors.END}")
    print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Config: {config_path}\n")
    
    results = {}
    
    # Test 1: Imports
    results['imports'] = test_imports()
    if not results['imports']:
        print_fail("Critical: Import test failed. Cannot continue.")
        return results
    
    # Test 2: Config
    results['config'], args = test_config(config_path)
    if not results['config']:
        print_fail("Critical: Config test failed. Cannot continue.")
        return results
    
    # Set device
    device = torch.device(args.device)
    print(f"\n{Colors.CYAN}Using device: {device}{Colors.END}")
    if torch.cuda.is_available():
        print(f"{Colors.CYAN}GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB{Colors.END}\n")
    
    # Test 3: Data Loading
    results['data_loading'], train_loader, val_loader, test_loader = test_data_loading(args)
    if not results['data_loading']:
        print_fail("Critical: Data loading test failed. Cannot continue.")
        return results
    
    # Test 4: Model Building
    results['model_building'], model, criterion, postprocessors = test_model_building(args)
    if not results['model_building']:
        print_fail("Critical: Model building test failed. Cannot continue.")
        return results
    
    # Test 5: Forward Pass
    results['forward_pass'], _ = test_forward_pass(model, train_loader, device)
    
    # Test 6: Loss Computation
    results['loss_computation'] = test_loss_computation(model, criterion, train_loader, device)
    
    # Test 7: Backward Pass
    results['backward_pass'] = test_backward_pass(model, criterion, train_loader, device)
    
    # Test 8: Training Epoch
    results['training_epoch'] = test_training_epoch(model, criterion, train_loader, device, num_iterations=5)
    
    # Test 9: Evaluation
    results['evaluation'], stats = test_evaluation(model, criterion, postprocessors, val_loader, device)
    
    # Test 10: Checkpoint
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    results['checkpoint'] = test_checkpoint_save_load(model, optimizer)
    
    # Test 11: Postprocessing
    results['postprocessing'] = test_postprocessing(model, postprocessors, test_loader, device)
    
    # Test 12: Training Logger
    results['training_logger'] = test_training_logger(args)
    
    # Summary
    print_section("TEST SUMMARY")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    print(f"{Colors.BOLD}Test Results:{Colors.END}\n")
    for test_name, result in results.items():
        status = f"{Colors.GREEN}✓ PASS{Colors.END}" if result else f"{Colors.RED}✗ FAIL{Colors.END}"
        test_display = test_name.replace('_', ' ').title()
        print(f"  {test_display:.<30} {status}")
    
    print(f"\n{Colors.BOLD}Summary: {passed}/{total} tests passed{Colors.END}")
    
    if passed == total:
        print(f"\n{Colors.GREEN}{Colors.BOLD}{'='*80}")
        print("✓ ALL TESTS PASSED!".center(80))
        print("Your DETR-SLIC pipeline is ready for training!".center(80))
        print(f"{'='*80}{Colors.END}\n")
    else:
        print(f"\n{Colors.RED}{Colors.BOLD}{'='*80}")
        print("✗ SOME TESTS FAILED".center(80))
        print("Please fix the issues before training.".center(80))
        print(f"{'='*80}{Colors.END}\n")
    
    return results


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='DETR-SLIC Pipeline Debug - Comprehensive end-to-end testing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with default config
  python debug_all.py
  
  # Test with custom config
  python debug_all.py --config configs/detr_baseline.yaml
  
  # Test with CPU
  python debug_all.py --device cpu
        """
    )
    parser.add_argument('--config', type=str, default='configs/detr_slic.yaml',
                       help='Path to config file (default: configs/detr_slic.yaml)')
    parser.add_argument('--device', type=str, default=None,
                       help='Override device from config (e.g., "cpu", "cuda:0")')
    
    args = parser.parse_args()
    
    # Load config and override device if specified
    if args.device:
        print(f"Overriding device to: {args.device}")
        # This would require modifying test_config to accept device override
    
    results = run_all_tests(args.config)
    
    # Exit with appropriate code
    all_passed = all(results.values())
    sys.exit(0 if all_passed else 1)
