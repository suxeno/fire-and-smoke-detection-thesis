# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import math
import os
import sys
import time
from typing import Iterable
from tqdm import tqdm

import torch
from torch.cuda.amp import autocast, GradScaler

import util.misc as utils
from datasets.coco_eval import CocoEvaluator


def _move_target_to_device(value, device):
    if isinstance(value, dict):
        moved = {}
        for k, v in value.items():
            if k == 'slic_maps' and isinstance(v, dict):
                # Keep full-resolution superpixel maps on CPU.
                # The model downsamples them before moving small maps to GPU.
                moved[k] = v
            else:
                moved[k] = _move_target_to_device(v, device)
        return moved
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    return value


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0,
                    amp: bool = False,
                    amp_dtype: str = 'fp16',
                    scaler: GradScaler | None = None,
                    eff_timing: bool = False,
                    eff_timing_sync_cuda: bool = False):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    pbar = tqdm(data_loader, desc=header, leave=False)

    use_amp = bool(amp) and device.type == 'cuda'
    if use_amp and amp_dtype not in ('fp16', 'bf16'):
        raise ValueError(f"Invalid amp_dtype={amp_dtype!r}. Choose 'fp16' or 'bf16'.")
    autocast_dtype = torch.float16 if amp_dtype == 'fp16' else torch.bfloat16
    use_scaler = use_amp and amp_dtype == 'fp16' and scaler is not None and scaler.is_enabled()

    for samples, targets in pbar:
        samples = samples.to(device)
        targets = [_move_target_to_device(t, device) for t in targets]

        iter_start = time.perf_counter() if eff_timing else None
        if eff_timing and eff_timing_sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        fwd_start = time.perf_counter() if eff_timing else None
        with autocast(enabled=use_amp, dtype=autocast_dtype):
            outputs = model(samples, targets)
        if eff_timing and eff_timing_sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        if eff_timing:
            fwd_ms = (time.perf_counter() - fwd_start) * 1000.0
            bs = int(samples.tensors.shape[0]) if hasattr(samples, 'tensors') else 0
            metric_logger.update(eff_forward_ms=fwd_ms)
            if fwd_ms > 0 and bs > 0:
                metric_logger.update(eff_imgs_per_s=(bs / (fwd_ms / 1000.0)))

        eff_out = {k: v for k, v in outputs.items() if isinstance(k, str) and k.startswith('eff_')}
        if eff_out:
            metric_logger.update(**eff_out)

        with autocast(enabled=use_amp, dtype=autocast_dtype):
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

        optimizer.zero_grad(set_to_none=True)
        if use_scaler:
            scaler.scale(losses).backward()
            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

        if eff_timing:
            iter_ms = (time.perf_counter() - iter_start) * 1000.0
            metric_logger.update(eff_iter_ms=iter_ms)

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        pbar.set_postfix({
            'loss': f"{loss_value:.4f}",
            'ce': f"{loss_dict_reduced_scaled['loss_ce'].item():.4f}"
        })

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir,
             amp: bool = False,
             amp_dtype: str = 'fp16',
             eff_timing: bool = False,
             eff_timing_sync_cuda: bool = False):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    pbar = tqdm(data_loader, desc=header, leave=False)

    use_amp = bool(amp) and device.type == 'cuda'
    if use_amp and amp_dtype not in ('fp16', 'bf16'):
        raise ValueError(f"Invalid amp_dtype={amp_dtype!r}. Choose 'fp16' or 'bf16'.")
    autocast_dtype = torch.float16 if amp_dtype == 'fp16' else torch.bfloat16

    def _to_fp32(obj):
        if torch.is_tensor(obj) and obj.is_floating_point():
            return obj.float()
        if isinstance(obj, dict):
            return {k: _to_fp32(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            t = [_to_fp32(v) for v in obj]
            return type(obj)(t)
        return obj

    for samples, targets in pbar:
        samples = samples.to(device)
        targets = [_move_target_to_device(t, device) for t in targets]

        if eff_timing and eff_timing_sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        fwd_start = time.perf_counter() if eff_timing else None
        with autocast(enabled=use_amp, dtype=autocast_dtype):
            outputs = model(samples, targets)
        if eff_timing and eff_timing_sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        if eff_timing:
            fwd_ms = (time.perf_counter() - fwd_start) * 1000.0
            bs = int(samples.tensors.shape[0]) if hasattr(samples, 'tensors') else 0
            metric_logger.update(eff_forward_ms=fwd_ms)
            if fwd_ms > 0 and bs > 0:
                metric_logger.update(eff_imgs_per_s=(bs / (fwd_ms / 1000.0)))

        eff_out = {k: v for k, v in outputs.items() if isinstance(k, str) and k.startswith('eff_')}
        if eff_out:
            metric_logger.update(**eff_out)

        with autocast(enabled=use_amp, dtype=autocast_dtype):
            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        pbar.set_postfix({
            'ce': f"{loss_dict_reduced_scaled['loss_ce'].item():.4f}"
        })

        outputs_fp32 = _to_fp32(outputs)
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs_fp32, orig_target_sizes)
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs_fp32, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
            
            # Extract custom metrics: per-category AP/Recall and mAP50
            per_category_stats = coco_evaluator.get_per_category_stats(iou_type='bbox')
            stats.update(per_category_stats)
            
            # Extract mAP50
            stats['mAP50'] = coco_evaluator.get_map_at_iou(iou_threshold=0.50, iou_type='bbox')
            
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    return stats, coco_evaluator

