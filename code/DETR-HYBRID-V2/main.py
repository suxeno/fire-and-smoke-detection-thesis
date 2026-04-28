# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler
import torch.nn as nn

import datasets
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch
from models import build_model


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    
    # DETR-HYBRID specific
    parser.add_argument('--slic_n_segments', default=200, type=int,
                        help="Number of superpixels to load and pool into.")
    parser.add_argument('--pooling_type', default='mean', type=str, choices=('mean', 'max', 'both'),
                        help="Type of pooling used per superpixel")
    parser.add_argument('--hybrid_token_mode', default='mixed', type=str, choices=('superpixel', 'mixed'),
                        help="Encoder token mode: superpixel-only or mixed pixel+superpixel tokens")
    parser.add_argument('--compact_superpixel_ids', action='store_true',
                        help="Compact superpixel ids per image to contiguous [0..K-1] before pooling")
    parser.add_argument('--require_superpixels', action='store_true',
                        help="Fail fast when expected precomputed superpixel files are missing")

    # Efficiency-first pixel-token pruning (DETR-HYBRID-V2)
    parser.add_argument('--pixel_prune', action='store_true',
                        help="Enable superpixel-driven pixel-token pruning for efficiency")
    parser.add_argument('--pixel_prune_keep_ratio', default=0.8, type=float,
                        help="Target keep ratio for pixel tokens (clamped to [0.6, 0.8])")
    parser.add_argument('--pixel_prune_score_mode', default='saliency', type=str,
                        choices=('saliency', 'feature_norm', 'counts'),
                        help="Superpixel ranking mode used to select kept pixel tokens")
    parser.add_argument('--pixel_prune_w_feature', default=0.45, type=float,
                        help="Weight for pooled feature-norm score in saliency ranking")
    parser.add_argument('--pixel_prune_w_color', default=0.25, type=float,
                        help="Weight for color saliency score in saliency ranking")
    parser.add_argument('--pixel_prune_w_texture', default=0.20, type=float,
                        help="Weight for texture/intensity score in saliency ranking")
    parser.add_argument('--pixel_prune_w_size', default=0.10, type=float,
                        help="Weight for size prior (log-count) in saliency ranking")

    # Efficiency benchmarking metrics (applies to both train and eval)
    parser.add_argument('--eff_timing', action='store_true',
                        help="Log forward-pass latency/throughput metrics")
    parser.add_argument('--eff_timing_sync_cuda', action='store_true',
                        help="Synchronize CUDA for accurate timings (slower)")

    # Mixed precision (AMP)
    parser.add_argument('--amp', action='store_true',
                        help='Enable native PyTorch AMP (autocast). Recommended on GPU.')
    parser.add_argument('--amp_dtype', default='fp16', type=str, choices=('fp16', 'bf16'),
                        help="AMP dtype: 'fp16' (uses GradScaler) or 'bf16' (no scaler).")

    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser


def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    device = torch.device(args.device)

    # AMP setup (only meaningful for CUDA)
    use_amp = bool(getattr(args, 'amp', False)) and device.type == 'cuda'
    if getattr(args, 'amp', False) and device.type != 'cuda':
        print("Warning: --amp requested but device is not CUDA; AMP will be disabled.")
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and args.amp_dtype == 'fp16'))

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessors = build_model(args)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    base_ds = get_coco_api_from_dataset(dataset_val)

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        
        # Filter out shape-mismatched keys (e.g. class_embed weight and bias for transfer learning)
        saved_state_dict = checkpoint['model']
        if saved_state_dict is not None:
             model_state_dict = model_without_ddp.state_dict()
             for k in list(saved_state_dict.keys()):
                 if k in model_state_dict:
                     if saved_state_dict[k].shape != model_state_dict[k].shape:
                         print(f"Removing key {k} from pretrained checkpoint due to shape mismatch")
                         del saved_state_dict[k]
        
        model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

    if args.eval:
        test_stats, coco_evaluator = evaluate(
            model,
            criterion,
            postprocessors,
            data_loader_val,
            base_ds,
            device,
            args.output_dir,
            amp=use_amp,
            amp_dtype=args.amp_dtype,
            eff_timing=args.eff_timing,
            eff_timing_sync_cuda=args.eff_timing_sync_cuda,
        )
        if args.output_dir:
            utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
        return

    # Initialize JSON log file (array of flat epoch records, same format as log.txt)
    json_log_path = output_dir / "training_log.json" if args.output_dir else None
    training_history = []  # Simple array of epoch records

    # Save experiment info for tracking
    if args.output_dir and utils.is_main_process():
        info_path = output_dir / "info.txt"
        with open(info_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("EXPERIMENT INFO\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("--- Timestamp ---\n")
            f.write(f"Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("--- Dataset ---\n")
            f.write(f"COCO Path: {args.coco_path}\n")
            f.write(f"Train samples: {len(dataset_train)}\n")
            f.write(f"Val samples: {len(dataset_val)}\n\n")

            f.write("--- Model Architecture ---\n")
            f.write(f"Backbone: {args.backbone}\n")
            f.write(f"Encoder layers: {args.enc_layers}\n")
            f.write(f"Decoder layers: {args.dec_layers}\n")
            f.write(f"Hidden dim: {args.hidden_dim}\n")
            f.write(f"Feedforward dim: {args.dim_feedforward}\n")
            f.write(f"Num heads: {args.nheads}\n")
            f.write(f"Num queries: {args.num_queries}\n")
            f.write(f"Dropout: {args.dropout}\n")
            f.write(f"Dilation: {args.dilation}\n")
            f.write(f"Position embedding: {args.position_embedding}\n")
            f.write(f"Auxiliary loss: {args.aux_loss}\n")
            f.write(f"Masks (segmentation): {args.masks}\n")
            f.write(f"Superpixel segments (DETR-HYBRID): {args.slic_n_segments}\n")
            f.write(f"Pooling type (DETR-HYBRID): {args.pooling_type}\n")
            f.write(f"Hybrid token mode (DETR-HYBRID): {args.hybrid_token_mode}\n")
            f.write(f"Compact superpixel ids (DETR-HYBRID): {args.compact_superpixel_ids}\n")
            f.write(f"Require superpixel files (DETR-HYBRID): {args.require_superpixels}\n")
            f.write(f"Pixel prune enabled (DETR-HYBRID-V2): {args.pixel_prune}\n")
            f.write(f"Pixel prune keep ratio (DETR-HYBRID-V2): {args.pixel_prune_keep_ratio}\n")
            f.write(f"Pixel prune score mode (DETR-HYBRID-V2): {args.pixel_prune_score_mode}\n")
            f.write(
                "Pixel prune weights (feature/color/texture/size) (DETR-HYBRID-V2): "
                f"{args.pixel_prune_w_feature}/{args.pixel_prune_w_color}/{args.pixel_prune_w_texture}/{args.pixel_prune_w_size}\n"
            )
            f.write(f"Efficiency timing enabled: {args.eff_timing}\n")
            f.write(f"Efficiency timing CUDA sync: {args.eff_timing_sync_cuda}\n")
            f.write(f"Total parameters: {n_parameters:,}\n\n")
            
            f.write("--- Training Config ---\n")
            f.write(f"Epochs: {args.epochs}\n")
            f.write(f"Start epoch: {args.start_epoch}\n")
            f.write(f"Batch size: {args.batch_size}\n")
            f.write(f"Learning rate: {args.lr}\n")
            f.write(f"LR backbone: {args.lr_backbone}\n")
            f.write(f"LR drop: {args.lr_drop}\n")
            f.write(f"Weight decay: {args.weight_decay}\n")
            f.write(f"Clip max norm: {args.clip_max_norm}\n\n")
            
            f.write("--- Loss Coefficients ---\n")
            f.write(f"Class cost: {args.set_cost_class}\n")
            f.write(f"BBox cost: {args.set_cost_bbox}\n")
            f.write(f"GIoU cost: {args.set_cost_giou}\n")
            f.write(f"BBox loss coef: {args.bbox_loss_coef}\n")
            f.write(f"GIoU loss coef: {args.giou_loss_coef}\n")
            f.write(f"EOS coef: {args.eos_coef}\n\n")
            
            f.write("--- Environment ---\n")
            f.write(f"Device: {args.device}\n")
            f.write(f"Seed: {args.seed}\n")
            f.write(f"Num workers: {args.num_workers}\n")
            f.write(f"Distributed: {args.distributed}\n")
            if args.resume:
                f.write(f"Resume from: {args.resume}\n")
            f.write(f"Output dir: {args.output_dir}\n\n")
            
            f.write("--- Git Info ---\n")
            f.write(f"{utils.get_sha()}\n")
            f.write("=" * 60 + "\n")
        print(f"Experiment info saved to {info_path}")

    print("Start training")
    start_time = time.time()

    # Initialize plotting early so output structure always includes plots/
    plots_dir = None
    plot_generator = None
    if args.output_dir and utils.is_main_process():
        plots_dir = output_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        try:
            from util.plot_utils import generate_training_plots
            plot_generator = generate_training_plots
        except Exception as e:
            print(f"Warning: Plot generator unavailable, plots will be skipped: {e}")
    
    # Track cumulative timing
    total_train_time = 0
    total_val_time = 0
    
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:    
            sampler_train.set_epoch(epoch)
        # Training with timing
        epoch_train_start = time.time()
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch,
            args.clip_max_norm,
            amp=use_amp,
            amp_dtype=args.amp_dtype,
            scaler=scaler,
            eff_timing=args.eff_timing,
            eff_timing_sync_cuda=args.eff_timing_sync_cuda,
        )
        train_time = time.time() - epoch_train_start
        total_train_time += train_time
        
        lr_scheduler.step()
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 100 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 100 == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

        # Validation with timing
        epoch_val_start = time.time()
        test_stats, coco_evaluator = evaluate(
            model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir
            ,
            amp=use_amp,
            amp_dtype=args.amp_dtype,
            eff_timing=args.eff_timing,
            eff_timing_sync_cuda=args.eff_timing_sync_cuda,
        )
        val_time = time.time() - epoch_val_start
        total_val_time += val_time

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters,
                     'train_time': train_time,
                     'val_time': val_time}

        # Add to history array
        training_history.append(log_stats)

        if args.output_dir and utils.is_main_process():
            # Save as proper JSON array (pretty printed)
            with open(json_log_path, 'w') as f:
                json.dump(training_history, f, indent=2)
            
            # Also keep line-by-line log.txt for compatibility
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

            # for evaluation logs
            if coco_evaluator is not None:
                (output_dir / 'eval').mkdir(exist_ok=True)
                if "bbox" in coco_evaluator.coco_eval:
                    filenames = ['latest.pth']
                    if epoch % 50 == 0:
                        filenames.append(f'{epoch:03}.pth')
                    for name in filenames:
                        torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                   output_dir / "eval" / name)

            # Update plots after each completed epoch so interrupted runs still keep plots
            if plot_generator is not None:
                try:
                    plot_generator(json_log_path, plots_dir)
                    print(f"Epoch {epoch}: training plots updated in {plots_dir}")
                except Exception as e:
                    print(f"Warning: Could not generate plots at epoch {epoch}: {e}")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    
    # Print timing summary
    num_epochs = args.epochs - args.start_epoch
    if num_epochs > 0:
        print("\n" + "=" * 50)
        print("TIMING SUMMARY")
        print("=" * 50)
        print(f"Total training time: {total_time_str}")
        print(f"Total TRAIN time: {str(datetime.timedelta(seconds=int(total_train_time)))}")
        print(f"Total VAL time: {str(datetime.timedelta(seconds=int(total_val_time)))}")
        print(f"Average TRAIN time per epoch: {str(datetime.timedelta(seconds=int(total_train_time / num_epochs)))}")
        print(f"Average VAL time per epoch: {str(datetime.timedelta(seconds=int(total_val_time / num_epochs)))}")
        print("=" * 50 + "\n")
    
    # Test set evaluation after training completes
    if args.output_dir and utils.is_main_process():
        print("\nRunning test set evaluation...")
        try:
            dataset_test = build_dataset(image_set='test', args=args)
            if args.distributed:
                sampler_test = DistributedSampler(dataset_test, shuffle=False)
            else:
                sampler_test = torch.utils.data.SequentialSampler(dataset_test)
            data_loader_test = DataLoader(dataset_test, args.batch_size, sampler=sampler_test,
                                          drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)
            base_ds_test = get_coco_api_from_dataset(dataset_test)
            
            test_start = time.time()
            test_stats_final, test_coco_evaluator = evaluate(
                model, criterion, postprocessors, data_loader_test, base_ds_test, device, args.output_dir
                ,
                amp=use_amp,
                amp_dtype=args.amp_dtype,
                eff_timing=args.eff_timing,
                eff_timing_sync_cuda=args.eff_timing_sync_cuda,
            )
            test_time = time.time() - test_start
            
            # Save test results
            test_log_path = output_dir / "test_log.json"
            test_log = {
                **{f'test_{k}': v for k, v in test_stats_final.items()},
                'test_time': test_time,
                'model_checkpoint': str(output_dir / 'checkpoint.pth'),
                'num_test_images': len(dataset_test)
            }
            with open(test_log_path, 'w') as f:
                json.dump(test_log, f, indent=2)
            print(f"Test set evaluation saved to {test_log_path}")
        except Exception as e:
            print(f"Warning: Could not run test set evaluation: {e}")
    
    # Final plot refresh after training completion
    if args.output_dir and utils.is_main_process() and plot_generator is not None:
        try:
            plot_generator(json_log_path, plots_dir)
            print(f"Final training plots saved to {plots_dir}")
        except Exception as e:
            print(f"Warning: Could not refresh final plots: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
