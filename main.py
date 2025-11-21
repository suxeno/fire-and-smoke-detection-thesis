"""
Main training script for DETR-SLIC Fire and Smoke Detection.
Uses YAML config files instead of argparse for easier configuration management.
"""
import argparse
import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

import util.misc as utils
from util import load_config, validate_config, build_data_loader
from engine import train_one_epoch, evaluate
from models import build_model


def main(config_path):
    """
    Main training function.
    
    Args:
        config_path: Path to YAML configuration file
    """
    # Load and validate configuration
    print(f"Loading config from: {config_path}")
    args = load_config(config_path)
    args = validate_config(args)
    
    # Initialize distributed training if needed
    if getattr(args, 'distributed', False):
        utils.init_distributed_mode(args)
    
    print("\n" + "="*80)
    print(f"Configuration for {args.model_name}")
    print("="*80)
    print(f"Model: {args.model_name}")
    print(f"Use SLIC: {args.use_slic}")
    print(f"Dataset: {args.data_path}")
    print(f"Num classes: {args.num_classes}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Output: {args.output_dir}")
    print("="*80 + "\n")
    
    device = torch.device(args.device)
    
    # Fix seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Build model, criterion, and postprocessors
    print("Building model...")
    model, criterion, postprocessors = build_model(args)
    model.to(device)
    
    # Model summary
    utils.print_model_summary(model, args.model_name)
    
    # Handle distributed training
    model_without_ddp = model
    if getattr(args, 'distributed', False):
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    
    # Build optimizer with different learning rates for backbone
    param_dicts = [
        {
            "params": [p for n, p in model_without_ddp.named_parameters() 
                      if "backbone" not in n and p.requires_grad]
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() 
                      if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
    
    # Build data loaders
    print("\nLoading datasets...")
    data_loader_train = build_data_loader('train', args)
    data_loader_val = build_data_loader('val', args)
    
    print(f"Training samples: {len(data_loader_train.dataset)}")
    print(f"Validation samples: {len(data_loader_val.dataset)}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config_save_path = output_dir / 'config.yaml'
    import shutil
    shutil.copy(config_path, config_save_path)
    print(f"Config saved to: {config_save_path}")
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            print(f"Resuming from epoch {args.start_epoch}")
    
    # Evaluation-only mode
    if args.eval:
        print("\n" + "="*80)
        print("EVALUATION MODE")
        print("="*80)
        test_stats = evaluate(
            model, criterion, postprocessors,
            data_loader_val, device, args.output_dir
        )
        print("\nEvaluation Results:")
        for k, v in test_stats.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, (int, float)) else f"  {k}: {v}")
        return
    
    # Training loop
    print("\n" + "="*80)
    print("STARTING TRAINING")
    print("="*80)
    start_time = time.time()
    best_val_loss = float('inf')
    
    for epoch in range(args.start_epoch, args.epochs):
        epoch_start_time = time.time()
        
        # Train one epoch
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer,
            device, epoch, args.clip_max_norm
        )
        lr_scheduler.step()
        
        # Validation
        if epoch % getattr(args, 'eval_every', 1) == 0:
            val_stats = evaluate(
                model, criterion, postprocessors,
                data_loader_val, device, args.output_dir
            )
        else:
            val_stats = {}
        
        # Logging
        epoch_time = time.time() - epoch_start_time
        log_stats = {
            **{f'train_{k}': v for k, v in train_stats.items()},
            **{f'val_{k}': v for k, v in val_stats.items()},
            'epoch': epoch,
            'epoch_time': epoch_time
        }
        
        # Print epoch summary
        print(f"\nEpoch {epoch}/{args.epochs} Summary:")
        print(f"  Train loss: {train_stats.get('loss', 0):.4f}")
        if val_stats:
            print(f"  Val loss: {val_stats.get('loss', 0):.4f}")
            if 'AP' in val_stats:
                print(f"  Val AP: {val_stats['AP']:.4f}")
                print(f"  Val AP50: {val_stats['AP50']:.4f}")
        print(f"  Time: {epoch_time:.1f}s")
        
        # Save log
        if utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
        
        # Save checkpoints
        if utils.is_main_process():
            checkpoint_dict = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': vars(args),
            }
            
            # Always save latest checkpoint
            torch.save(checkpoint_dict, output_dir / 'checkpoint_latest.pth')
            
            # Save best checkpoint based on validation loss
            if val_stats and val_stats.get('loss', float('inf')) < best_val_loss:
                best_val_loss = val_stats['loss']
                torch.save(checkpoint_dict, output_dir / 'checkpoint_best.pth')
                print(f"  ✓ Saved best checkpoint (val_loss: {best_val_loss:.4f})")
            
            # Save periodic checkpoints
            if hasattr(args, 'save_every') and (epoch + 1) % args.save_every == 0:
                torch.save(checkpoint_dict, output_dir / f'checkpoint_epoch_{epoch:04d}.pth')
                print(f"  ✓ Saved checkpoint at epoch {epoch}")
    
    # Training complete
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"Total time: {total_time_str}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Results saved to: {output_dir}")
    print("="*80 + "\n")
    
    # Run final test evaluation
    print("Running final evaluation on test set...")
    try:
        data_loader_test = build_data_loader('test', args)
        test_stats = evaluate(
            model, criterion, postprocessors,
            data_loader_test, device, args.output_dir
        )
        
        print("\nFinal Test Results:")
        for k, v in test_stats.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, (int, float)) else f"  {k}: {v}")
        
        # Save test results
        if utils.is_main_process():
            with (output_dir / "test_results.json").open("w") as f:
                json.dump(test_stats, f, indent=2)
    except Exception as e:
        print(f"⚠ Test evaluation skipped: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR-SLIC training script')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to YAML config file (e.g., configs/detr_slic.yaml)')
    cmd_args = parser.parse_args()
    
    main(cmd_args.config)
