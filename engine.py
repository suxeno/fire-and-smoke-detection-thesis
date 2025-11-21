"""
Train and eval functions used in main.py
Adapted for Detection with DETR-SLIC
"""
import math
import sys
import time 
from typing import Iterable

import torch

import util.misc as utils

# Optional COCO evaluation (if pycocotools is available)
try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    HAS_COCO_EVAL = True
except ImportError:
    HAS_COCO_EVAL = False
    print("pycocotools not available. COCO evaluation will be skipped.")


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 50  # Print every 50 iterations instead of 10

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        # Only track key metrics (not all auxiliary losses)
        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_ce=loss_dict_reduced_scaled['loss_ce'])
        metric_logger.update(loss_bbox=loss_dict_reduced_scaled['loss_bbox'])
        metric_logger.update(loss_giou=loss_dict_reduced_scaled['loss_giou'])
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, device, output_dir=None):
    """
    Evaluate model on validation/test set.
    
    Args:
        model: DETR or DETR-SLIC model
        criterion: Loss criterion
        postprocessors: Dict with 'bbox' postprocessor
        data_loader: DataLoader for evaluation
        device: torch.device
        output_dir: Optional output directory for COCO eval results
    
    Returns:
        stats: Dict with evaluation metrics
    """
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Val:' if 'val' in str(data_loader.dataset.ann_file).lower() else 'Test:'

    # Collect predictions for optional COCO evaluation
    coco_results = []
    
    for samples, targets in metric_logger.log_every(data_loader, 50, header):  # Print every 50 iterations
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        
        # Only track main metrics (not auxiliary losses)
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()))
        metric_logger.update(loss_ce=loss_dict_reduced_scaled['loss_ce'])
        metric_logger.update(loss_bbox=loss_dict_reduced_scaled['loss_bbox'])
        metric_logger.update(loss_giou=loss_dict_reduced_scaled['loss_giou'])
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        # Postprocess predictions
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        
        # Store results for COCO evaluation
        for target, result in zip(targets, results):
            image_id = target['image_id'].item()
            boxes = result['boxes'].cpu()
            scores = result['scores'].cpu()
            labels = result['labels'].cpu()
            
            # Convert to COCO format
            for box, score, label in zip(boxes, scores, labels):
                x, y, x2, y2 = box.tolist()
                w, h = x2 - x, y2 - y
                coco_results.append({
                    'image_id': image_id,
                    'category_id': label.item(),
                    'bbox': [x, y, w, h],
                    'score': score.item()
                })

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    
    # Optional COCO evaluation
    if HAS_COCO_EVAL and len(coco_results) > 0:
        try:
            coco_eval_stats = evaluate_coco(
                data_loader.dataset.ann_file,
                coco_results
            )
            stats.update(coco_eval_stats)
            
            # Print formatted evaluation results
            print("\n" + "="*80)
            print("COCO Evaluation Results:")
            print(f"  AP      (IoU=0.50:0.95): {coco_eval_stats.get('AP', 0):.4f}")
            print(f"  mAP50   (IoU=0.50):      {coco_eval_stats.get('mAP50', 0):.4f}")
            print(f"  AP75    (IoU=0.75):      {coco_eval_stats.get('AP75', 0):.4f}")
            print(f"  Recall  (Overall):       {coco_eval_stats.get('Recall', 0):.4f}")
            print()
            print("Per-Class Metrics:")
            # Display fire metrics if available
            if 'Recall_fire' in coco_eval_stats:
                print(f"  Fire    - AP: {coco_eval_stats.get('AP_fire', 0):.4f} | AP50: {coco_eval_stats.get('AP50_fire', 0):.4f} | Recall: {coco_eval_stats.get('Recall_fire', 0):.4f}")
            # Display smoke metrics if available
            if 'Recall_smoke' in coco_eval_stats:
                print(f"  Smoke   - AP: {coco_eval_stats.get('AP_smoke', 0):.4f} | AP50: {coco_eval_stats.get('AP50_smoke', 0):.4f} | Recall: {coco_eval_stats.get('Recall_smoke', 0):.4f}")
            print("="*80 + "\n")
        except Exception as e:
            print(f"⚠ COCO evaluation failed: {e}")
    
    return stats


def evaluate_coco(ann_file, coco_results):
    """
    Run COCO evaluation if pycocotools is available.
    
    Args:
        ann_file: Path to COCO annotation file
        coco_results: List of detection results in COCO format
    
    Returns:
        Dict with COCO metrics
    """
    if not HAS_COCO_EVAL:
        return {}
    
    import json
    import tempfile
    from pathlib import Path
    
    # Load ground truth
    coco_gt = COCO(str(ann_file))
    
    # Fix missing 'info' field in dataset (common issue with custom COCO datasets)
    if 'info' not in coco_gt.dataset:
        coco_gt.dataset['info'] = {
            'description': 'Fire and Smoke Detection Dataset',
            'version': '1.0',
            'year': 2024,
            'contributor': 'DETR-SLIC',
            'date_created': '2024/01/01'
        }
    
    # Load predictions
    coco_dt = coco_gt.loadRes(coco_results)
    
    # Run evaluation
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    # Extract standard metrics
    stats = {
        'coco_eval_bbox': coco_eval.stats.tolist(),
        'AP': coco_eval.stats[0],           # AP @ IoU=0.50:0.95
        'AP50': coco_eval.stats[1],         # AP @ IoU=0.50
        'mAP50': coco_eval.stats[1],        # Same as AP50 (mean AP at IoU=0.50)
        'AP75': coco_eval.stats[2],         # AP @ IoU=0.75
        'AP_small': coco_eval.stats[3],
        'AP_medium': coco_eval.stats[4],
        'AP_large': coco_eval.stats[5],
        'AR_max1': coco_eval.stats[6],      # AR @ 1 detection
        'AR_max10': coco_eval.stats[7],     # AR @ 10 detections
        'AR_max100': coco_eval.stats[8],    # AR @ 100 detections (Recall)
        'Recall': coco_eval.stats[8],       # Overall Recall @ IoU=0.50:0.95
        'AR_small': coco_eval.stats[9],
        'AR_medium': coco_eval.stats[10],
        'AR_large': coco_eval.stats[11],
    }
    
    # Per-class metrics (fire=0, smoke=1)
    # precision has dims (iou_thresholds, recall_thresholds, categories, area_ranges, max_dets)
    # recall has dims (iou_thresholds, categories, area_ranges, max_dets)
    
    # Get category IDs (should be 0 and 1 for fire and smoke)
    cat_ids = sorted(coco_gt.getCatIds())
    
    # Compute per-class AP and Recall
    for i, cat_id in enumerate(cat_ids):
        # AP per class: average over all IoU thresholds
        # precision shape: (iou, recall_thresh, category, area, max_det)
        ap_per_class = coco_eval.eval['precision'][:, :, i, 0, 2]  # All IoU, all recall, cat i, all areas, max_det=100
        ap_per_class = ap_per_class[ap_per_class > -1]  # Remove -1 values
        ap_class = ap_per_class.mean() if len(ap_per_class) > 0 else 0.0
        
        # AP50 per class: IoU=0.50 (index 0)
        ap50_per_class = coco_eval.eval['precision'][0, :, i, 0, 2]  # IoU=0.50, all recall, cat i
        ap50_per_class = ap50_per_class[ap50_per_class > -1]
        ap50_class = ap50_per_class.mean() if len(ap50_per_class) > 0 else 0.0
        
        # Recall per class: recall shape (iou, category, area, max_det)
        recall_per_class = coco_eval.eval['recall'][:, i, 0, 2]  # All IoU, cat i, all areas, max_det=100
        recall_per_class = recall_per_class[recall_per_class > -1]
        recall_class = recall_per_class.mean() if len(recall_per_class) > 0 else 0.0
        
        # Recall @ IoU=0.50 per class
        recall50_per_class = coco_eval.eval['recall'][0, i, 0, 2]  # IoU=0.50, cat i
        recall50_class = recall50_per_class if recall50_per_class > -1 else 0.0
        
        # Get category name
        cat_info = coco_gt.loadCats([cat_id])[0]
        cat_name = cat_info['name']
        
        # Store per-class metrics
        stats[f'AP_{cat_name}'] = float(ap_class)
        stats[f'AP50_{cat_name}'] = float(ap50_class)
        stats[f'Recall_{cat_name}'] = float(recall_class)
        stats[f'Recall50_{cat_name}'] = float(recall50_class)
    
    return stats