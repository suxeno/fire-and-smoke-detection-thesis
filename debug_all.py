#!/usr/bin/env python3
"""
Comprehensive Debug Script for DETR and DETR-SLIC Pipelines
===========================================================
Tests all components from data loading to training, validation, and testing.
Supports testing both DETR baseline and DETR-SLIC architectures.

Usage:
    # Test DETR-SLIC (default)
    python debug_all.py
    
    # Test DETR baseline
    python debug_all.py --config configs/detr_baseline.yaml
    
    # Test both models
    python debug_all.py --both
"""
import argparse
import sys
import os
import tempfile
import traceback
from pathlib import Path
from itertools import islice
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


# =============================================================================
# Terminal Colors
# =============================================================================
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


def print_test(msg):
    print(f"{Colors.BLUE}[TEST]{Colors.END} {msg}")

def print_pass(msg):
    print(f"{Colors.GREEN}[PASS]{Colors.END} ✓ {msg}")

def print_fail(msg):
    print(f"{Colors.RED}[FAIL]{Colors.END} ✗ {msg}")

def print_warn(msg):
    print(f"{Colors.YELLOW}[WARN]{Colors.END} ⚠ {msg}")

def print_info(msg):
    print(f"{Colors.CYAN}[INFO]{Colors.END} ℹ {msg}")

def print_section(msg):
    print(f"\n{Colors.BOLD}{'='*80}")
    print(f"{msg}")
    print(f"{'='*80}{Colors.END}\n")


# =============================================================================
# Test Functions
# =============================================================================

def test_imports():
    """Test 1: Verify all required imports work"""
    print_section("TEST 1: Checking Imports")
    
    try:
        print_test("Importing PyTorch...")
        import torch
        import torchvision
        print_pass(f"PyTorch {torch.__version__}, Torchvision {torchvision.__version__}")
        print_info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print_info(f"CUDA version: {torch.version.cuda}")
            print_info(f"GPU: {torch.cuda.get_device_name(0)}")
        
        print_test("Importing project modules...")
        from util import load_config, validate_config, build_data_loader
        from util.training_logger import TrainingLogger
        from engine import train_one_epoch, evaluate
        from models import build_model
        print_pass("Core project modules imported successfully")
        
        print_test("Checking optional dependencies...")
        try:
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval
            print_pass("pycocotools available (COCO metrics enabled)")
        except ImportError:
            print_warn("pycocotools not available (COCO metrics will be disabled)")
        
        try:
            import matplotlib.pyplot as plt
            print_pass("matplotlib available")
        except ImportError:
            print_warn("matplotlib not available")
        
        try:
            from skimage.segmentation import slic
            print_pass("scikit-image available (for SLIC generation)")
        except ImportError:
            print_warn("scikit-image not available (pre-computed superpixels required)")
        
        return True
    except Exception as e:
        print_fail(f"Import test failed: {e}")
        traceback.print_exc()
        return False


def test_config(config_path):
    """Test 2: Load and validate configuration"""
    print_section("TEST 2: Config Loading and Validation")
    
    try:
        from util import load_config, validate_config
        
        print_test(f"Loading config from {config_path}...")
        if not Path(config_path).exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        args = load_config(config_path)
        print_pass("Config loaded successfully")
        
        print_test("Validating config...")
        args = validate_config(args)
        print_pass("Config validated")
        
        print_test("Critical config parameters:")
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
        
        if args.use_slic:
            print(f"  N superpixels:     {getattr(args, 'n_superpixels', 100)}")
            print(f"  SLIC compactness:  {getattr(args, 'slic_compactness', 10.0)}")
            print(f"  Use SCA:           {getattr(args, 'use_sca', False)}")
        
        # Verify dataset paths exist
        print_test("Verifying dataset paths...")
        data_path = Path(args.data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {data_path}")
        print_pass(f"Dataset path exists: {data_path}")
        
        images_path = data_path / 'images'
        if not images_path.exists():
            raise FileNotFoundError(f"Images directory not found: {images_path}")
        print_pass(f"Images directory exists")
        
        ann_path = data_path / 'annotations' / 'COCO' / 'Annotations'
        if not ann_path.exists():
            raise FileNotFoundError(f"Annotations directory not found: {ann_path}")
        print_pass(f"Annotations directory exists")
        
        # Check superpixels if SLIC is enabled
        if args.use_slic:
            sp_path = data_path / 'superpixels'
            if not sp_path.exists():
                print_warn(f"Superpixels directory not found: {sp_path}")
                print_warn("Run 'python util/generate_superpixel.py' to generate superpixels")
            else:
                print_pass(f"Superpixels directory exists")
        
        return True, args
    except Exception as e:
        print_fail(f"Config test failed: {e}")
        traceback.print_exc()
        return False, None


def test_data_loading(args):
    """Test 3: Data loading pipeline"""
    print_section("TEST 3: Data Loading Pipeline")
    
    try:
        from util import build_data_loader
        
        # Load train dataset
        print_test("Loading training dataset...")
        train_loader = build_data_loader('train', args)
        print_pass(f"Train loader: {len(train_loader.dataset)} samples, {len(train_loader)} batches")
        
        # Load validation dataset
        print_test("Loading validation dataset...")
        val_loader = build_data_loader('val', args)
        print_pass(f"Val loader: {len(val_loader.dataset)} samples, {len(val_loader)} batches")
        
        # Test batch loading
        print_test("Testing batch loading...")
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
        print_pass(f"Target keys: {list(target.keys())}")
        
        # Check superpixel maps if SLIC enabled
        if args.use_slic:
            if 'slic_map' in target:
                sp_map = target['slic_map']
                print_pass(f"Superpixel map found: shape={sp_map.shape}, max_id={sp_map.max()}")
            else:
                print_warn("Superpixel map NOT found in target - run generate_superpixel.py first!")
        
        # Validate target values
        print_test("Validating target values...")
        print_info(f"  Number of objects: {len(target['boxes'])}")
        if len(target['boxes']) > 0:
            print_info(f"  Boxes shape: {target['boxes'].shape}")
            print_info(f"  Labels: {target['labels'].tolist()}")
            
            # Verify labels are within valid range
            max_label = target['labels'].max().item()
            if max_label >= args.num_classes:
                raise ValueError(f"Label {max_label} >= num_classes {args.num_classes}")
            print_pass("All labels within valid range")
        else:
            print_warn("This sample has no annotations (background image)")
        
        return True, train_loader, val_loader
    except Exception as e:
        print_fail(f"Data loading test failed: {e}")
        traceback.print_exc()
        return False, None, None


def test_model_building(args):
    """Test 4: Model building and architecture verification"""
    print_section("TEST 4: Model Building")
    
    try:
        from models import build_model
        from util.misc import count_parameters
        
        print_test("Building model...")
        model, criterion, postprocessors = build_model(args)
        print_pass(f"Model built: {type(model).__name__}")
        
        # Count parameters
        print_test("Counting parameters...")
        param_counts = count_parameters(model)
        total_params = param_counts['total']
        trainable_params = param_counts['trainable']
        print_pass(f"Total parameters: {total_params:,}")
        print_info(f"  Trainable: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")
        
        # Verify model components
        print_test("Checking model architecture...")
        
        if hasattr(model, 'backbone'):
            print_pass(f"  Backbone: {model.backbone.num_channels} channels")
        
        if args.use_slic:
            # DETR-SLIC specific components
            if hasattr(model, 'superpixel_pool'):
                print_pass(f"  SuperpixelPool: max={model.superpixel_pool.max_superpixels}")
            if hasattr(model, 'superpixel_encoder'):
                print_pass(f"  SuperpixelEncoder: {model.superpixel_encoder.d_model}d")
            if hasattr(model, 'pos_embed'):
                print_pass(f"  Position Embedding: {type(model.pos_embed).__name__}")
            if hasattr(model, 'sca') and model.use_sca:
                print_pass(f"  SCA: enabled")
            if hasattr(model, 'decoder'):
                print_pass(f"  Decoder: {type(model.decoder).__name__}")
        else:
            # Standard DETR
            if hasattr(model, 'transformer'):
                print_pass(f"  Transformer: {model.transformer.d_model}d")
        
        if hasattr(model, 'class_embed'):
            out_features = model.class_embed.out_features
            print_pass(f"  Class head: {out_features} outputs ({out_features-1} classes + 1 no-object)")
        
        if hasattr(model, 'bbox_embed'):
            print_pass(f"  Box head: MLP with {model.bbox_embed.num_layers} layers")
        
        # Verify criterion
        print_test("Checking criterion...")
        print_pass(f"Criterion: {type(criterion).__name__}")
        print_info(f"  num_classes: {criterion.num_classes}")
        print_info(f"  eos_coef: {criterion.eos_coef}")
        print_info(f"  Loss weights: {criterion.weight_dict}")
        
        return True, model, criterion, postprocessors
    except Exception as e:
        print_fail(f"Model building test failed: {e}")
        traceback.print_exc()
        return False, None, None, None


def test_forward_pass(model, train_loader, args, device):
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
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        
        print_test("Running forward pass...")
        with torch.no_grad():
            if args.use_slic:
                outputs = model(samples, targets)  # DETR-SLIC needs targets for slic_map
            else:
                outputs = model(samples)  # Standard DETR doesn't need targets
        print_pass("Forward pass completed")
        
        # Verify output format
        print_test("Checking output format...")
        assert 'pred_logits' in outputs, "Missing pred_logits"
        assert 'pred_boxes' in outputs, "Missing pred_boxes"
        
        pred_logits = outputs['pred_logits']
        pred_boxes = outputs['pred_boxes']
        
        print_info(f"  pred_logits: {pred_logits.shape}")
        print_info(f"  pred_boxes: {pred_boxes.shape}")
        
        # Verify shapes
        assert pred_logits.shape[0] == batch_size
        assert pred_boxes.shape[0] == batch_size
        assert pred_logits.shape[1] == args.num_queries
        assert pred_boxes.shape[1] == args.num_queries
        assert pred_logits.shape[2] == args.num_classes + 1  # classes + no-object
        assert pred_boxes.shape[2] == 4
        print_pass("Output shapes verified")
        
        # Check for NaN/Inf
        if torch.isnan(pred_logits).any() or torch.isinf(pred_logits).any():
            raise ValueError("NaN or Inf in pred_logits!")
        if torch.isnan(pred_boxes).any() or torch.isinf(pred_boxes).any():
            raise ValueError("NaN or Inf in pred_boxes!")
        print_pass("No NaN or Inf values")
        
        # Analyze predictions
        probs = pred_logits.softmax(-1)
        print_info(f"  Class probs - Fire: {probs[:,:,0].mean():.4f}, Smoke: {probs[:,:,1].mean():.4f}, No-obj: {probs[:,:,2].mean():.4f}")
        print_info(f"  Box range: [{pred_boxes.min():.4f}, {pred_boxes.max():.4f}]")
        
        return True, outputs
    except Exception as e:
        print_fail(f"Forward pass test failed: {e}")
        traceback.print_exc()
        return False, None


def test_loss_computation(model, criterion, train_loader, args, device):
    """Test 6: Loss computation"""
    print_section("TEST 6: Loss Computation")
    
    try:
        model = model.to(device)
        model.train()
        
        print_test("Getting a batch...")
        samples, targets = next(iter(train_loader))
        samples = samples.to(device)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        
        print_test("Computing forward pass...")
        if args.use_slic:
            outputs = model(samples, targets)
        else:
            outputs = model(samples)
        
        print_test("Computing losses...")
        loss_dict = criterion(outputs, targets)
        print_pass("Losses computed")
        
        print_test("Checking loss components...")
        required_losses = ['loss_ce', 'loss_bbox', 'loss_giou']
        for loss_name in required_losses:
            if loss_name not in loss_dict:
                raise ValueError(f"Missing loss: {loss_name}")
            loss_val = loss_dict[loss_name].item()
            if not (0 <= loss_val < 1000):
                print_warn(f"  {loss_name}: {loss_val:.4f} (unusual value)")
            else:
                print_info(f"  {loss_name}: {loss_val:.4f}")
        
        # Compute total loss
        weight_dict = criterion.weight_dict
        total_loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        print_pass(f"Total weighted loss: {total_loss.item():.4f}")
        
        return True, loss_dict
    except Exception as e:
        print_fail(f"Loss computation test failed: {e}")
        traceback.print_exc()
        return False, None


def test_backward_pass(model, criterion, train_loader, args, device):
    """Test 7: Backward pass and gradient computation"""
    print_section("TEST 7: Backward Pass and Gradients")
    
    try:
        model = model.to(device)
        model.train()
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        
        print_test("Getting a batch...")
        samples, targets = next(iter(train_loader))
        samples = samples.to(device)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        
        print_test("Forward pass...")
        if args.use_slic:
            outputs = model(samples, targets)
        else:
            outputs = model(samples)
        
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        total_loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        print_info(f"  Total loss: {total_loss.item():.4f}")
        
        print_test("Backward pass...")
        optimizer.zero_grad()
        total_loss.backward()
        print_pass("Gradients computed")
        
        print_test("Checking gradients...")
        grad_count = 0
        grad_norms = []
        zero_grad_params = []
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_count += 1
                norm = param.grad.norm().item()
                grad_norms.append(norm)
                if norm == 0:
                    zero_grad_params.append(name)
        
        avg_grad_norm = sum(grad_norms) / len(grad_norms) if grad_norms else 0
        max_grad_norm = max(grad_norms) if grad_norms else 0
        
        print_pass(f"Gradients verified: {grad_count} parameters")
        print_info(f"  Avg gradient norm: {avg_grad_norm:.6f}")
        print_info(f"  Max gradient norm: {max_grad_norm:.6f}")
        
        if zero_grad_params:
            print_warn(f"  {len(zero_grad_params)} parameters have zero gradients")
        
        if max_grad_norm > 100:
            print_warn("Large gradients detected - gradient clipping recommended")
        
        print_test("Applying gradient clipping...")
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
        print_pass("Gradients clipped (max_norm=0.1)")
        
        print_test("Optimizer step...")
        optimizer.step()
        print_pass("Weights updated")
        
        return True
    except Exception as e:
        print_fail(f"Backward pass test failed: {e}")
        traceback.print_exc()
        return False


def test_training_iterations(model, criterion, train_loader, args, device, num_iterations=5):
    """Test 8: Multiple training iterations"""
    print_section(f"TEST 8: Training ({num_iterations} iterations)")
    
    try:
        model = model.to(device)
        model.train()
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        
        print_test(f"Training for {num_iterations} iterations...")
        
        losses_history = []
        
        for i, (samples, targets) in enumerate(islice(train_loader, num_iterations)):
            samples = samples.to(device)
            targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
            
            if args.use_slic:
                outputs = model(samples, targets)
            else:
                outputs = model(samples)
            
            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict
            total_loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
            
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            optimizer.step()
            
            losses_history.append(total_loss.item())
            print_info(f"  Iter {i+1}: loss={total_loss.item():.4f}")
        
        print_pass("Training iterations completed")
        
        # Check loss trend
        if len(losses_history) > 1:
            first_half = sum(losses_history[:len(losses_history)//2]) / (len(losses_history)//2)
            second_half = sum(losses_history[len(losses_history)//2:]) / (len(losses_history) - len(losses_history)//2)
            
            if second_half < first_half:
                print_pass(f"Loss is decreasing: {first_half:.4f} → {second_half:.4f}")
            else:
                print_info(f"Loss trend: {first_half:.4f} → {second_half:.4f} (normal for few iterations)")
        
        return True
    except Exception as e:
        print_fail(f"Training iterations test failed: {e}")
        traceback.print_exc()
        return False


def test_evaluation(model, criterion, postprocessors, val_loader, args, device):
    """Test 9: Evaluation"""
    print_section("TEST 9: Evaluation")
    
    try:
        from engine import evaluate
        
        print_test("Running evaluation (limited batches)...")
        
        # Create a limited loader for quick testing
        class LimitedLoader:
            def __init__(self, loader, limit=5):
                self.loader = loader
                self.limit = limit
                self.dataset = loader.dataset
            def __iter__(self):
                return islice(iter(self.loader), self.limit)
            def __len__(self):
                return min(len(self.loader), self.limit)
        
        limited_val = LimitedLoader(val_loader, limit=5)
        
        stats = evaluate(model, criterion, postprocessors, limited_val, device)
        print_pass("Evaluation completed")
        
        print_test("Checking evaluation metrics...")
        required_metrics = ['loss', 'loss_ce', 'loss_bbox', 'loss_giou']
        for metric in required_metrics:
            if metric in stats:
                print_info(f"  {metric}: {stats[metric]:.4f}")
        
        if 'AP' in stats:
            print_info(f"  AP (mAP): {stats['AP']:.4f}")
            print_info(f"  AP50: {stats.get('AP50', stats.get('mAP50', 'N/A'))}")
            print_info(f"  Recall: {stats.get('Recall', 'N/A')}")
        else:
            print_warn("COCO metrics not available (pycocotools may not be installed)")
        
        return True, stats
    except Exception as e:
        print_fail(f"Evaluation test failed: {e}")
        traceback.print_exc()
        return False, None


def test_checkpoint(model, args):
    """Test 10: Checkpoint save/load"""
    print_section("TEST 10: Checkpoint Save/Load")
    
    try:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        print_test("Creating checkpoint...")
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': 0,
            'args': vars(args) if hasattr(args, '__dict__') else dict(args),
        }
        
        print_test("Saving checkpoint...")
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            checkpoint_path = f.name
            torch.save(checkpoint, f)
        
        checkpoint_size = os.path.getsize(checkpoint_path) / (1024 * 1024)
        print_pass(f"Checkpoint saved ({checkpoint_size:.2f} MB)")
        
        print_test("Loading checkpoint...")
        loaded = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(loaded['model'])
        optimizer.load_state_dict(loaded['optimizer'])
        print_pass("Checkpoint loaded successfully")
        
        print_test("Cleaning up...")
        os.remove(checkpoint_path)
        print_pass("Temporary checkpoint deleted")
        
        return True
    except Exception as e:
        print_fail(f"Checkpoint test failed: {e}")
        traceback.print_exc()
        return False


def test_postprocessing(model, postprocessors, val_loader, args, device):
    """Test 11: Postprocessing predictions"""
    print_section("TEST 11: Postprocessing")
    
    try:
        model = model.to(device)
        model.eval()
        
        print_test("Getting test batch...")
        samples, targets = next(iter(val_loader))
        samples = samples.to(device)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        
        print_test("Running inference...")
        with torch.no_grad():
            if args.use_slic:
                outputs = model(samples, targets)
            else:
                outputs = model(samples)
        
        print_test("Postprocessing outputs...")
        orig_sizes = torch.stack([t["orig_size"] for t in targets], dim=0).to(device)
        results = postprocessors['bbox'](outputs, orig_sizes)
        print_pass(f"Postprocessed {len(results)} predictions")
        
        print_test("Checking result format...")
        result = results[0]
        required_keys = ['scores', 'labels', 'boxes']
        for key in required_keys:
            if key not in result:
                raise ValueError(f"Missing key: {key}")
        print_pass(f"Result keys: {list(result.keys())}")
        
        # Analyze predictions
        print_test("Analyzing predictions...")
        scores = result['scores']
        labels = result['labels']
        boxes = result['boxes']
        
        # Filter by confidence threshold
        threshold = 0.5
        high_conf = scores > threshold
        n_high_conf = high_conf.sum().item()
        
        print_info(f"  Total predictions: {len(scores)}")
        print_info(f"  High confidence (>{threshold}): {n_high_conf}")
        
        if n_high_conf > 0:
            filtered_labels = labels[high_conf]
            unique_labels, counts = filtered_labels.unique(return_counts=True)
            for label, count in zip(unique_labels.tolist(), counts.tolist()):
                class_name = ["fire", "smoke"][label] if label < 2 else "unknown"
                print_info(f"    {class_name}: {count}")
        
        return True
    except Exception as e:
        print_fail(f"Postprocessing test failed: {e}")
        traceback.print_exc()
        return False


# =============================================================================
# Main Test Runner
# =============================================================================

def run_all_tests(config_path):
    """Run all tests for a given configuration"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}")
    print("="*80)
    print("DETR PIPELINE DEBUG".center(80))
    print("="*80)
    print(f"{Colors.END}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Config: {config_path}\n")
    
    results = {}
    
    # Test 1: Imports
    results['imports'] = test_imports()
    if not results['imports']:
        print_fail("Critical: Import test failed")
        return results, None
    
    # Test 2: Config
    results['config'], args = test_config(config_path)
    if not results['config']:
        print_fail("Critical: Config test failed")
        return results, None
    
    # Set device
    device = torch.device(args.device)
    print(f"\n{Colors.CYAN}Using device: {device}{Colors.END}")
    if torch.cuda.is_available():
        print(f"{Colors.CYAN}GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB{Colors.END}\n")
    
    # Test 3: Data Loading
    results['data_loading'], train_loader, val_loader = test_data_loading(args)
    if not results['data_loading']:
        print_fail("Critical: Data loading failed")
        return results, args
    
    # Test 4: Model Building
    results['model_building'], model, criterion, postprocessors = test_model_building(args)
    if not results['model_building']:
        print_fail("Critical: Model building failed")
        return results, args
    
    # Test 5: Forward Pass
    results['forward_pass'], _ = test_forward_pass(model, train_loader, args, device)
    
    # Test 6: Loss Computation
    results['loss_computation'], _ = test_loss_computation(model, criterion, train_loader, args, device)
    
    # Test 7: Backward Pass
    results['backward_pass'] = test_backward_pass(model, criterion, train_loader, args, device)
    
    # Test 8: Training Iterations
    results['training_iterations'] = test_training_iterations(model, criterion, train_loader, args, device)
    
    # Test 9: Evaluation
    results['evaluation'], _ = test_evaluation(model, criterion, postprocessors, val_loader, args, device)
    
    # Test 10: Checkpoint
    results['checkpoint'] = test_checkpoint(model, args)
    
    # Test 11: Postprocessing
    results['postprocessing'] = test_postprocessing(model, postprocessors, val_loader, args, device)
    
    # Summary
    print_section("TEST SUMMARY")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    print(f"{Colors.BOLD}Results for {args.model_name}:{Colors.END}\n")
    for test_name, result in results.items():
        status = f"{Colors.GREEN}✓ PASS{Colors.END}" if result else f"{Colors.RED}✗ FAIL{Colors.END}"
        test_display = test_name.replace('_', ' ').title()
        print(f"  {test_display:.<35} {status}")
    
    print(f"\n{Colors.BOLD}Summary: {passed}/{total} tests passed{Colors.END}")
    
    if passed == total:
        print(f"\n{Colors.GREEN}{Colors.BOLD}{'='*80}")
        print(f"✓ ALL TESTS PASSED for {args.model_name}!".center(80))
        print(f"{'='*80}{Colors.END}\n")
    else:
        print(f"\n{Colors.RED}{Colors.BOLD}{'='*80}")
        print(f"✗ SOME TESTS FAILED for {args.model_name}".center(80))
        print(f"{'='*80}{Colors.END}\n")
    
    return results, args


def main():
    parser = argparse.ArgumentParser(
        description='DETR Pipeline Debug - Comprehensive end-to-end testing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test DETR-SLIC (default)
  python debug_all.py
  
  # Test DETR baseline
  python debug_all.py --config configs/detr_baseline.yaml
  
  # Test both models
  python debug_all.py --both
        """
    )
    parser.add_argument('--config', type=str, default='configs/detr_slic.yaml',
                       help='Path to config file')
    parser.add_argument('--both', action='store_true',
                       help='Test both DETR baseline and DETR-SLIC')
    
    args = parser.parse_args()
    
    configs_to_test = []
    
    if args.both:
        configs_to_test = [
            'configs/detr_baseline.yaml',
            'configs/detr_slic.yaml'
        ]
    else:
        configs_to_test = [args.config]
    
    all_results = {}
    
    for config_path in configs_to_test:
        if not Path(config_path).exists():
            print_warn(f"Config not found: {config_path}, skipping...")
            continue
        
        results, tested_args = run_all_tests(config_path)
        if tested_args:
            all_results[tested_args.model_name] = results
    
    # Final summary if testing multiple
    if len(all_results) > 1:
        print_section("FINAL SUMMARY - ALL MODELS")
        for model_name, results in all_results.items():
            passed = sum(1 for v in results.values() if v)
            total = len(results)
            status = f"{Colors.GREEN}✓{Colors.END}" if passed == total else f"{Colors.RED}✗{Colors.END}"
            print(f"  {model_name}: {status} {passed}/{total} tests passed")
        
        all_passed = all(
            all(results.values()) for results in all_results.values()
        )
        
        if all_passed:
            print(f"\n{Colors.GREEN}{Colors.BOLD}ALL MODELS PASSED!{Colors.END}")
            return 0
        else:
            print(f"\n{Colors.RED}{Colors.BOLD}SOME MODELS HAVE FAILURES{Colors.END}")
            return 1
    
    # Single model result
    if all_results:
        first_result = list(all_results.values())[0]
        return 0 if all(first_result.values()) else 1
    
    return 1


if __name__ == '__main__':
    sys.exit(main())
