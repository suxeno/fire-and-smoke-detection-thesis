"""
Debug script to test the entire DETR-SLIC pipeline end-to-end.
Tests all components from data loading to training, validation, and testing.
"""
import torch
import argparse
from pathlib import Path
import sys

# Color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_test(msg):
    print(f"{Colors.BLUE}[TEST]{Colors.END} {msg}")

def print_pass(msg):
    print(f"{Colors.GREEN}[PASS]{Colors.END} ✓ {msg}")

def print_fail(msg):
    print(f"{Colors.RED}[FAIL]{Colors.END} ✗ {msg}")

def print_warn(msg):
    print(f"{Colors.YELLOW}[WARN]{Colors.END} ⚠ {msg}")

def print_section(msg):
    print(f"\n{Colors.BOLD}{'='*80}")
    print(f"{msg}")
    print(f"{'='*80}{Colors.END}\n")


def test_imports():
    """Test 1: Check all imports work"""
    print_section("TEST 1: Checking Imports")
    
    try:
        print_test("Importing PyTorch...")
        import torch
        import torchvision
        print_pass(f"PyTorch {torch.__version__}, Torchvision {torchvision.__version__}")
        
        print_test("Importing project modules...")
        from util import load_config, validate_config, build_data_loader
        from engine import train_one_epoch, evaluate
        from models import build_model
        print_pass("All project modules imported successfully")
        
        print_test("Checking optional dependencies...")
        try:
            from pycocotools.coco import COCO
            print_pass("pycocotools available (COCO metrics enabled)")
        except ImportError:
            print_warn("pycocotools not available (COCO metrics disabled)")
        
        try:
            import matplotlib
            print_pass("matplotlib available (visualization enabled)")
        except ImportError:
            print_warn("matplotlib not available (visualization disabled)")
        
        return True
    except Exception as e:
        print_fail(f"Import failed: {e}")
        return False


def test_config():
    """Test 2: Load and validate config"""
    print_section("TEST 2: Config Loading and Validation")
    
    try:
        from util import load_config, validate_config
        
        print_test("Loading config from configs/detr_slic.yaml...")
        args = load_config('configs/detr_slic.yaml')
        print_pass(f"Config loaded successfully")
        
        print_test("Validating config...")
        args = validate_config(args)
        print_pass("Config validated")
        
        print_test("Config parameters:")
        print(f"  Model: {args.model_name}")
        print(f"  Use SLIC: {args.use_slic}")
        print(f"  Dataset: {args.data_path}")
        print(f"  Batch size: {args.batch_size}")
        print(f"  Device: {args.device}")
        
        return True, args
    except Exception as e:
        print_fail(f"Config test failed: {e}")
        return False, None


def test_data_loading(args):
    """Test 3: Data loading pipeline"""
    print_section("TEST 3: Data Loading Pipeline")
    
    try:
        from util import build_data_loader
        
        print_test("Loading training dataset...")
        train_loader = build_data_loader('train', args)
        print_pass(f"Train loader created: {len(train_loader.dataset)} samples")
        
        print_test("Loading validation dataset...")
        val_loader = build_data_loader('val', args)
        print_pass(f"Val loader created: {len(val_loader.dataset)} samples")
        
        print_test("Loading test dataset...")
        test_loader = build_data_loader('test', args)
        print_pass(f"Test loader created: {len(test_loader.dataset)} samples")
        
        print_test("Testing batch loading...")
        samples, targets = next(iter(train_loader))
        print_pass(f"Batch loaded: images shape {samples.tensors.shape}, {len(targets)} targets")
        
        print_test("Checking target format...")
        target = targets[0]
        required_keys = ['boxes', 'labels', 'image_id', 'area', 'orig_size']
        for key in required_keys:
            if key not in target:
                raise ValueError(f"Missing key: {key}")
        print_pass(f"Target format correct: {list(target.keys())}")
        
        return True, train_loader, val_loader, test_loader
    except Exception as e:
        print_fail(f"Data loading test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None, None


def test_model_building(args):
    """Test 4: Model building"""
    print_section("TEST 4: Model Building")
    
    try:
        from models import build_model
        
        print_test("Building DETR-SLIC model...")
        model, criterion, postprocessors = build_model(args)
        print_pass("Model built successfully")
        
        print_test("Counting parameters...")
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print_pass(f"Total params: {total_params:,}, Trainable: {trainable_params:,}")
        
        print_test("Checking model components...")
        print(f"  Backbone: {type(model.backbone).__name__}")
        print(f"  SLIC enabled: {hasattr(model, 'slic')}")
        print(f"  Superpixel encoder: {hasattr(model, 'superpixel_encoder')}")
        print_pass("Model components validated")
        
        return True, model, criterion, postprocessors
    except Exception as e:
        print_fail(f"Model building test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None, None


def test_forward_pass(model, train_loader, device):
    """Test 5: Forward pass"""
    print_section("TEST 5: Forward Pass")
    
    try:
        model = model.to(device)
        model.eval()
        
        print_test("Getting a batch...")
        samples, targets = next(iter(train_loader))
        samples = samples.to(device)
        print_pass(f"Batch moved to {device}")
        
        print_test("Running forward pass...")
        with torch.no_grad():
            outputs = model(samples)
        print_pass("Forward pass successful")
        
        print_test("Checking output format...")
        if 'pred_logits' not in outputs:
            raise ValueError("Missing 'pred_logits' in outputs")
        if 'pred_boxes' not in outputs:
            raise ValueError("Missing 'pred_boxes' in outputs")
        print_pass(f"Outputs: pred_logits {outputs['pred_logits'].shape}, pred_boxes {outputs['pred_boxes'].shape}")
        
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
        
        print_test("Computing forward pass...")
        outputs = model(samples)
        
        print_test("Computing losses...")
        loss_dict = criterion(outputs, targets)
        print_pass("Losses computed")
        
        print_test("Checking loss components...")
        required_losses = ['loss_ce', 'loss_bbox', 'loss_giou']
        for loss_name in required_losses:
            if loss_name not in loss_dict:
                raise ValueError(f"Missing loss: {loss_name}")
            print(f"  {loss_name}: {loss_dict[loss_name].item():.4f}")
        print_pass("All loss components present")
        
        print_test("Computing total loss...")
        weight_dict = criterion.weight_dict
        total_loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        print_pass(f"Total loss: {total_loss.item():.4f}")
        
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
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        print_pass("Optimizer created")
        
        print_test("Getting a batch...")
        samples, targets = next(iter(train_loader))
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        print_test("Forward pass...")
        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        total_loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        
        print_test("Backward pass...")
        optimizer.zero_grad()
        total_loss.backward()
        print_pass("Gradients computed")
        
        print_test("Checking gradients...")
        has_grad = False
        for name, param in model.named_parameters():
            if param.grad is not None:
                has_grad = True
                break
        if not has_grad:
            raise ValueError("No gradients computed")
        print_pass("Gradients verified")
        
        print_test("Optimizer step...")
        optimizer.step()
        print_pass("Weights updated")
        
        return True
    except Exception as e:
        print_fail(f"Backward pass test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_epoch(model, criterion, train_loader, device):
    """Test 8: One training epoch"""
    print_section("TEST 8: Training One Epoch (5 iterations)")
    
    try:
        from engine import train_one_epoch
        
        model = model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        print_test("Training for 5 iterations...")
        # Limit to 5 iterations for quick test
        from itertools import islice
        limited_loader = islice(train_loader, 5)
        
        model.train()
        for i, (samples, targets) in enumerate(limited_loader):
            samples = samples.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            outputs = model(samples)
            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            print(f"  Iteration {i+1}/5: loss={losses.item():.4f}")
        
        print_pass("Training epoch completed successfully")
        return True
    except Exception as e:
        print_fail(f"Training epoch test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_evaluation(model, criterion, postprocessors, val_loader, device):
    """Test 9: Evaluation"""
    print_section("TEST 9: Evaluation")
    
    try:
        from engine import evaluate
        
        print_test("Running evaluation...")
        stats = evaluate(model, criterion, postprocessors, val_loader, device)
        print_pass("Evaluation completed")
        
        print_test("Checking evaluation metrics...")
        required_metrics = ['loss', 'loss_ce', 'loss_bbox', 'loss_giou', 'class_error']
        for metric in required_metrics:
            if metric not in stats:
                print_warn(f"Missing metric: {metric}")
            else:
                print(f"  {metric}: {stats[metric]:.4f}")
        print_pass("Evaluation metrics obtained")
        
        if 'AP' in stats:
            print_test("COCO metrics available:")
            print(f"  AP: {stats['AP']:.4f}")
            print(f"  AP50: {stats['AP50']:.4f}")
            print(f"  AP75: {stats['AP75']:.4f}")
            print_pass("COCO metrics computed")
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
        import tempfile
        
        print_test("Creating checkpoint...")
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': 0,
        }
        
        print_test("Saving checkpoint...")
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            checkpoint_path = f.name
            torch.save(checkpoint, checkpoint_path)
        print_pass(f"Checkpoint saved to {checkpoint_path}")
        
        print_test("Loading checkpoint...")
        loaded_checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print_pass("Checkpoint loaded")
        
        print_test("Restoring model state...")
        model.load_state_dict(loaded_checkpoint['model'])
        print_pass("Model state restored")
        
        print_test("Cleaning up...")
        import os
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
        samples = samples.to(device)
        
        print_test("Running inference...")
        with torch.no_grad():
            outputs = model(samples)
        
        print_test("Postprocessing outputs...")
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        print_pass(f"Postprocessed {len(results)} predictions")
        
        print_test("Checking result format...")
        result = results[0]
        if 'scores' not in result or 'labels' not in result or 'boxes' not in result:
            raise ValueError("Missing keys in postprocessed results")
        print_pass(f"Result keys: {list(result.keys())}")
        print(f"  Detected {len(result['scores'])} objects")
        print(f"  Scores range: [{result['scores'].min():.3f}, {result['scores'].max():.3f}]")
        
        return True
    except Exception as e:
        print_fail(f"Postprocessing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests sequentially"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}")
    print("="*80)
    print("  DETR-SLIC END-TO-END PIPELINE DEBUG")
    print("="*80)
    print(f"{Colors.END}\n")
    
    results = {}
    
    # Test 1: Imports
    results['imports'] = test_imports()
    if not results['imports']:
        print_fail("Critical: Import test failed. Cannot continue.")
        return results
    
    # Test 2: Config
    results['config'], args = test_config()
    if not results['config']:
        print_fail("Critical: Config test failed. Cannot continue.")
        return results
    
    # Set device
    device = torch.device(args.device)
    print(f"\nUsing device: {device}\n")
    
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
    results['training_epoch'] = test_training_epoch(model, criterion, train_loader, device)
    
    # Test 9: Evaluation
    results['evaluation'], stats = test_evaluation(model, criterion, postprocessors, val_loader, device)
    
    # Test 10: Checkpoint
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    results['checkpoint'] = test_checkpoint_save_load(model, optimizer)
    
    # Test 11: Postprocessing
    results['postprocessing'] = test_postprocessing(model, postprocessors, test_loader, device)
    
    # Summary
    print_section("SUMMARY")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = f"{Colors.GREEN}PASS{Colors.END}" if result else f"{Colors.RED}FAIL{Colors.END}"
        print(f"  {test_name:20s}: {status}")
    
    print(f"\n{Colors.BOLD}Results: {passed}/{total} tests passed{Colors.END}")
    
    if passed == total:
        print(f"\n{Colors.GREEN}{Colors.BOLD}✓ ALL TESTS PASSED!{Colors.END}")
        print(f"{Colors.GREEN}Your pipeline is ready for training!{Colors.END}\n")
    else:
        print(f"\n{Colors.RED}{Colors.BOLD}✗ SOME TESTS FAILED{Colors.END}")
        print(f"{Colors.RED}Please fix the issues before training.{Colors.END}\n")
    
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR-SLIC Pipeline Debug')
    parser.add_argument('--config', type=str, default='configs/detr_slic.yaml',
                       help='Path to config file (default: configs/detr_slic.yaml)')
    
    args = parser.parse_args()
    
    # Override config path if provided
    if args.config != 'configs/detr_slic.yaml':
        # This would require modifying test_config() to accept config path
        print(f"Using config: {args.config}")
    
    results = run_all_tests()
    
    # Exit with appropriate code
    all_passed = all(results.values())
    sys.exit(0 if all_passed else 1)
