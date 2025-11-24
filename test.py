"""
Test/Evaluation script for DETR-SLIC Fire and Smoke Detection.
Loads trained model and evaluates on test set with optional visualization.
"""
import argparse
import json
from pathlib import Path

import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np

from util import load_config, validate_config, build_data_loader
from engine import evaluate
from models import build_model


def visualize_predictions(image, target, predictions, output_path, class_names=['fire', 'smoke'], threshold=0.5):
    """
    Visualize ground truth and predictions side by side.
    
    Args:
        image: PIL Image
        target: Ground truth dict with 'boxes' and 'labels'
        predictions: Prediction dict with 'boxes', 'scores', 'labels'
        output_path: Where to save visualization
        class_names: List of class names
        threshold: Confidence threshold for displaying predictions
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Convert image to numpy for display
    img_np = np.array(image)
    
    # Ground truth
    ax1.imshow(img_np)
    ax1.set_title('Ground Truth', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    if target is not None and 'boxes' in target:
        boxes_gt = target['boxes'].cpu().numpy()
        labels_gt = target['labels'].cpu().numpy()
        
        for box, label in zip(boxes_gt, labels_gt):
            x, y, x2, y2 = box
            w, h = x2 - x, y2 - y
            color = 'red' if label == 0 else 'yellow'  # fire=red, smoke=yellow
            rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor='none')
            ax1.add_patch(rect)
            ax1.text(x, y - 5, class_names[label], color=color, fontsize=10, 
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor=color, boxstyle='round,pad=0.3'))
    
    # Predictions
    ax2.imshow(img_np)
    ax2.set_title(f'Predictions (conf > {threshold})', fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    if predictions is not None:
        boxes_pred = predictions['boxes'].cpu().numpy()
        scores_pred = predictions['scores'].cpu().numpy()
        labels_pred = predictions['labels'].cpu().numpy()
        
        for box, score, label in zip(boxes_pred, scores_pred, labels_pred):
            if score >= threshold:
                x, y, x2, y2 = box
                w, h = x2 - x, y2 - y
                color = 'red' if label == 0 else 'yellow'
                rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor='none')
                ax2.add_patch(rect)
                ax2.text(x, y - 5, f'{class_names[label]} {score:.2f}', color=color, fontsize=10,
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor=color, boxstyle='round,pad=0.3'))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def test_model(config_path, checkpoint_path, visualize=False, num_visualize=10, threshold=0.5):
    """
    Test trained model on test set.
    
    Args:
        config_path: Path to YAML config file
        checkpoint_path: Path to trained checkpoint
        visualize: Whether to save visualizations
        num_visualize: Number of images to visualize
        threshold: Confidence threshold for visualization
    """
    print(f"Loading config from: {config_path}")
    args = load_config(config_path)
    args = validate_config(args)
    args.eval = True  # Set to evaluation mode
    
    device = torch.device(args.device)
    
    print("\n" + "="*100)
    print("DETR-SLIC TEST EVALUATION".center(100))
    print("="*100)
    print(f"Model:      {args.model_name}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Dataset:    {args.data_path}")
    print("="*100 + "\n")
    
    # Build model
    print("Building model...")
    model, criterion, postprocessors = build_model(args)
    
    # Load checkpoint
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    
    print(f"✓ Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    
    # Build test data loader
    print("\nLoading test datasets...")
    test_categories = ['CV', 'UAV', 'RS']
    data_loaders_test = {}
    for cat in test_categories:
        print(f"Building test loader for {cat}...")
        data_loaders_test[cat] = build_data_loader('test', args, filter_category=cat)
    
    # Create output directory
    output_dir = Path(args.output_dir) / 'test_results'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Logs directory
    logs_dir = Path('outputs/logs')
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    if visualize:
        viz_dir = output_dir / 'visualizations'
        viz_dir.mkdir(exist_ok=True)
        print(f"Visualizations will be saved to: {viz_dir}")
    
    # Run evaluation
    print("\nRunning evaluation on test set...")
    
    all_test_stats = {}
    
    for cat, loader in data_loaders_test.items():
        print(f"\nTesting on {cat}...")
        if len(loader.dataset) == 0:
            print(f"⚠ No samples found for {cat}, skipping.")
            continue
            
        test_stats = evaluate(
            model, criterion, postprocessors,
            loader, device, str(output_dir)
        )
        
        # Store stats with category prefix
        for k, v in test_stats.items():
            all_test_stats[f'{cat}_{k}'] = v
            
        # Print results with clean formatting
        print("\n" + "="*100)
        print(f"TEST SET RESULTS ({cat})".center(100))
        print("="*100)
        
        # Loss metrics
        print(f"\n{'LOSS METRICS:':<20} Loss: {test_stats.get('loss', 0):.4f} | " +
              f"Class Error: {test_stats.get('class_error', 0):.2f}%")
        if test_stats.get('loss_ce') is not None:
            print(f"{'':20} CE: {test_stats.get('loss_ce', 0):.4f} | " +
                  f"BBox: {test_stats.get('loss_bbox', 0):.4f} | " +
                  f"GIoU: {test_stats.get('loss_giou', 0):.4f}")
        
        # COCO metrics (if available)
        if 'AP' in test_stats:
            print(f"\n{'DETECTION METRICS:':<20} mAP: {test_stats.get('AP', 0):.4f} | " +
                  f"mAP50: {test_stats.get('mAP50', test_stats.get('AP50', 0)):.4f} | " +
                  f"Recall: {test_stats.get('Recall', 0):.4f}")
            print(f"{'':20} AP75: {test_stats.get('AP75', 0):.4f} | " +
                  f"AP_small: {test_stats.get('AP_small', 0):.4f} | " +
                  f"AP_medium: {test_stats.get('AP_medium', 0):.4f} | " +
                  f"AP_large: {test_stats.get('AP_large', 0):.4f}")
            
            # Per-class metrics
            if 'AP_fire' in test_stats or 'AP_smoke' in test_stats:
                print(f"\n{'PER-CLASS METRICS:':<20}")
                if 'AP_fire' in test_stats:
                    print(f"  {'Fire:':<18} AP: {test_stats.get('AP_fire', 0):.4f} | " +
                          f"AP50: {test_stats.get('AP50_fire', 0):.4f} | " +
                          f"Recall: {test_stats.get('Recall_fire', 0):.4f}")
                if 'AP_smoke' in test_stats:
                    print(f"  {'Smoke:':<18} AP: {test_stats.get('AP_smoke', 0):.4f} | " +
                          f"AP50: {test_stats.get('AP50_smoke', 0):.4f} | " +
                          f"Recall: {test_stats.get('Recall_smoke', 0):.4f}")
    
    print("\n" + "="*100 + "\n")
    
    # Save results to JSON in logs directory
    results_file = logs_dir / 'test_results.json'
    with open(results_file, 'w') as f:
        json.dump(all_test_stats, f, indent=2)
    print(f"✓ Results saved to: {results_file}")
    
    # Visualize predictions (using first available loader)
    if visualize:
        print(f"\nGenerating {num_visualize} visualizations...")
        model.eval()
        
        class_names = ['fire', 'smoke']
        viz_count = 0
        
        # Use combined loader or iterate through categories
        # For simplicity, just visualize from the first non-empty category or iterate
        
        with torch.no_grad():
            for cat, loader in data_loaders_test.items():
                if viz_count >= num_visualize:
                    break
                
                for samples, targets in loader:
                    if viz_count >= num_visualize:
                        break
                    
                    # Move to device
                    samples = samples.to(device)
                    
                    # Get predictions
                    outputs = model(samples)
                    
                    # Postprocess
                    orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
                    results = postprocessors['bbox'](outputs, orig_target_sizes)
                    
                    # Visualize each image in batch
                    for i, (target, result) in enumerate(zip(targets, results)):
                        if viz_count >= num_visualize:
                            break
                        
                        # Get original image
                        img_id = target['image_id'].item()
                        img_info = loader.dataset.images[img_id]
                        img_path = loader.dataset.img_folder / img_info['file_name']
                        image = Image.open(img_path).convert('RGB')
                        
                        # Save visualization
                        viz_path = viz_dir / f'test_{cat}_{viz_count:04d}_id_{img_id}.png'
                        visualize_predictions(image, target, result, viz_path, class_names, threshold)
                        
                        viz_count += 1
                        if (viz_count) % 5 == 0:
                            print(f"  Generated {viz_count}/{num_visualize} visualizations")
        
        print(f"✓ Saved {viz_count} visualizations to: {viz_dir}")
    
    print("\n" + "="*100)
    print("TEST EVALUATION COMPLETE".center(100))
    print("="*100)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR-SLIC test/evaluation script')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to YAML config file (e.g., configs/detr_slic.yaml)')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained checkpoint (e.g., outputs/detr_slic/checkpoint_best.pth)')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualization of predictions')
    parser.add_argument('--num-visualize', type=int, default=10,
                       help='Number of images to visualize (default: 10)')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Confidence threshold for visualization (default: 0.5)')
    
    args = parser.parse_args()
    
    test_model(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        visualize=args.visualize,
        num_visualize=args.num_visualize,
        threshold=args.threshold
    )
