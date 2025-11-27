"""
Main training script for DETR-SLIC Fire and Smoke Detection.
Uses YAML config files instead of argparse for easier configuration management.

Per-subset training: Train and evaluate on a single subset (CV, UAV, or RS).
Example: python main.py --config configs/detr_baseline.yaml --subset CV
"""
import argparse
import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

import util.misc as utils
from util import load_config, validate_config, build_data_loader
from util.training_logger import TrainingLogger
from util.plot_utils import plot_logs
from engine import train_one_epoch, evaluate
from models import build_model


def main(config_path, subset=None):
    """
    Main training function.
    
    Args:
        config_path: Path to YAML configuration file
        subset: Optional subset to train/eval on (CV, UAV, RS). If None, uses config value or trains on all data.
    """
    # Load and validate configuration
    print(f"Loading config from: {config_path}")
    args = load_config(config_path)
    args = validate_config(args)
    
    # Handle subset filtering
    # Priority: CLI argument > config file > None (all data)
    filter_category = subset or getattr(args, 'subset', None)
    
    # Update output directory to include subset if specified
    if filter_category:
        base_output = getattr(args, 'output_dir', 'outputs/default')
        args.output_dir = f"{base_output}/{filter_category.lower()}"
        args.model_name = f"{args.model_name}_{filter_category.lower()}"
    
    # Initialize distributed training if needed
    if getattr(args, 'distributed', False):
        utils.init_distributed_mode(args)
    
    print("\n" + "="*80)
    print(f"Configuration for {args.model_name}")
    print("="*80)
    print(f"Model: {args.model_name}")
    print(f"Use SLIC: {args.use_slic}")
    print(f"Dataset: {args.data_path}")
    print(f"Subset: {filter_category if filter_category else 'ALL'}")
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
    data_loader_train = build_data_loader('train', args, filter_category=filter_category)
    data_loader_val = build_data_loader('val', args, filter_category=filter_category)
    
    print(f"Training samples: {len(data_loader_train.dataset)}")
    print(f"Validation samples: {len(data_loader_val.dataset)}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create plots and logs directories
    plots_dir = Path('outputs/plots')
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    logs_dir = Path('outputs/logs')
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize clean logger
    logger = TrainingLogger(output_dir, args.model_name)
    
    # Save configuration
    config_save_path = output_dir / 'config.yaml'
    import shutil
    shutil.copy(config_path, config_save_path)
    print(f"Config saved to: {config_save_path}")
    
    # Also copy config to logs directory for centralized tracking
    shutil.copy(config_path, logs_dir / f'{args.model_name}_config.yaml')
    
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
        
        print(f"\nEvaluating on {'subset ' + filter_category if filter_category else 'all data'}:")
        test_stats = evaluate(
            model, criterion, postprocessors,
            data_loader_val, device, args.output_dir
        )
        print(f"\nEvaluation Results:")
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
        val_stats = {}
        if epoch % getattr(args, 'eval_every', 1) == 0:
            print(f"\nValidating...")
            val_stats = evaluate(
                model, criterion, postprocessors,
                data_loader_val, device, args.output_dir
            )
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        
        # Use clean logger for epoch summary
        logger.log_epoch(epoch, train_stats, val_stats, epoch_time)
        
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
    
    # Training complete
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    
    # Run final test evaluation
    test_stats = {}
    try:
        print("\nRunning final evaluation on test set...")
        data_loader_test = build_data_loader('test', args, filter_category=filter_category)
        
        print(f"Test samples: {len(data_loader_test.dataset)}")
        test_stats = evaluate(
            model, criterion, postprocessors,
            data_loader_test, device, args.output_dir
        )
        
        # Save test results
        if utils.is_main_process():
            test_results_path = output_dir / "test_results.json"
            with test_results_path.open("w") as f:
                json.dump(test_stats, f, indent=2)
            print(f"Test results saved to: {test_results_path}")
    except Exception as e:
        print(f"⚠ Test evaluation skipped: {e}")
    
    # Log final results
    logger.log_final_results(total_time_str, test_stats)
    
    # Generate training plots
    if utils.is_main_process():
        try:
            print("Generating training plots...")
            plot_logs(
                logs=[logs_dir],  # Use logs_dir where log.txt is actually saved
                fields=('loss', 'mAP', 'mAP50', 'Recall'),
                ewm_col=0,
                log_name='log.txt'
            )
            plt.savefig(plots_dir / f'{args.model_name}_training_curves.png', dpi=150, bbox_inches='tight')
            plt.close()
            print(f"✓ Training plots saved to: {plots_dir / f'{args.model_name}_training_curves.png'}")
        except Exception as e:
            print(f"⚠ Plot generation failed: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR-SLIC training script')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to YAML config file (e.g., configs/detr_slic.yaml)')
    parser.add_argument('--subset', type=str, choices=['CV', 'UAV', 'RS'], default=None,
                       help='Subset to train on (CV, UAV, or RS). Default: train on all data.')
    cmd_args = parser.parse_args()
    
    main(cmd_args.config, cmd_args.subset)
