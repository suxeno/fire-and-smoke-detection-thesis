"""
Plotting utilities to visualize training logs.
"""
import torch
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path, PurePath


def plot_logs(logs, fields=('class_error', 'loss_bbox_unscaled', 'mAP'), ewm_col=0, log_name='log.txt'):
    '''
    Function to plot specific fields from training log(s). Plots both training and test results.

    :: Inputs - logs = list containing Path objects, each pointing to individual dir with a log file
              - fields = which results to plot from each log file - plots both training and test for each field.
              - ewm_col = optional, which column to use as the exponential weighted smoothing of the plots
              - log_name = optional, name of log file if different than default 'log.txt'.

    :: Outputs - matplotlib plots of results in fields, color coded for each log file.
               - solid lines are training results, dashed lines are test results.

    '''
    func_name = "plot_utils.py::plot_logs"

    # verify logs is a list of Paths (list[Paths]) or single Pathlib object Path,
    # convert single Path to list to avoid 'not iterable' error

    if not isinstance(logs, list):
        if isinstance(logs, PurePath):
            logs = [logs]
            print(f"{func_name} info: logs param expects a list argument, converted to list[Path].")
        else:
            raise ValueError(f"{func_name} - invalid argument for logs parameter.\n \
            Expect list[Path] or single Path obj, received {type(logs)}")

    # Quality checks - verify valid dir(s), that every item in list is Path object, and that log_name exists in each dir
    for i, dir in enumerate(logs):
        if not isinstance(dir, PurePath):
            raise ValueError(f"{func_name} - non-Path object in logs argument of {type(dir)}: \n{dir}")
        if not dir.exists():
            raise ValueError(f"{func_name} - invalid directory in logs argument:\n{dir}")
        # verify log_name exists
        fn = Path(dir / log_name)
        if not fn.exists():
            print(f"-> missing {log_name}.  Have you gotten to Epoch 1 in training?")
            print(f"--> full path of missing log file: {fn}")
            return

    # load log file(s) and plot
    dfs = [pd.read_json(Path(p) / log_name, lines=True) for p in logs]

    fig, axs = plt.subplots(ncols=len(fields), figsize=(16, 5))

    for df, color in zip(dfs, sns.color_palette(n_colors=len(logs))):
        for j, field in enumerate(fields):
            if field == 'mAP':
                coco_eval = pd.DataFrame(
                    np.stack(df.test_coco_eval_bbox.dropna().values)[:, 1]
                ).ewm(com=ewm_col).mean()
                axs[j].plot(coco_eval, c=color)
            else:
                df.interpolate().ewm(com=ewm_col).mean().plot(
                    y=[f'train_{field}', f'test_{field}'],
                    ax=axs[j],
                    color=[color] * 2,
                    style=['-', '--']
                )
    for ax, field in zip(axs, fields):
        ax.legend([Path(p).name for p in logs])
        ax.set_title(field)


def plot_precision_recall(files, naming_scheme='iter'):
    if naming_scheme == 'exp_id':
        # name becomes exp_id
        names = [f.parts[-3] for f in files]
    elif naming_scheme == 'iter':
        names = [f.stem for f in files]
    else:
        raise ValueError(f'not supported {naming_scheme}')
    fig, axs = plt.subplots(ncols=2, figsize=(16, 5))
    for f, color, name in zip(files, sns.color_palette("Blues", n_colors=len(files)), names):
        data = torch.load(f)
        # precision is n_iou, n_points, n_cat, n_area, max_det
        precision = data['precision']
        recall = data['params'].recThrs
        scores = data['scores']
        # take precision for all classes, all areas and 100 detections
        precision = precision[0, :, :, 0, -1].mean(1)
        scores = scores[0, :, :, 0, -1].mean(1)
        prec = precision.mean()
        rec = data['recall'][0, :, 0, -1].mean()
        print(f'{naming_scheme} {name}: mAP@50={prec * 100: 05.1f}, ' +
              f'score={scores.mean():0.3f}, ' +
              f'f1={2 * prec * rec / (prec + rec + 1e-8):0.3f}'
              )
        axs[0].plot(recall, precision, c=color)
        axs[1].plot(recall, scores, c=color)

    axs[0].set_title('Precision / Recall')
    axs[0].legend(names)
    axs[1].set_title('Scores / Recall')
    axs[1].legend(names)
    return fig, axs


def generate_training_plots(log_path, output_dir):
    """
    Generate comprehensive training plots from training_log.json.
    
    Args:
        log_path: Path to training_log.json
        output_dir: Directory to save plots
    
    Generates:
        - loss_curves.png: Training and validation losses
        - ap_metrics.png: mAP50, AP_fire, AP_smoke
        - recall_metrics.png: Recall_fire, Recall_smoke, AR@100
        - timing.png: Train and validation time per epoch
    """
    from pathlib import Path
    import json
    
    log_path = Path(log_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load training log
    with open(log_path, 'r') as f:
        history = json.load(f)
    
    if not history:
        print("Warning: Empty training log, no plots generated")
        return
    
    df = pd.DataFrame(history)
    epochs = df['epoch'].values
    
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # 1. Loss Curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Total loss
    if 'train_loss' in df.columns:
        axes[0].plot(epochs, df['train_loss'], 'b-', label='Train Loss', linewidth=2)
    if 'test_loss' in df.columns:
        axes[0].plot(epochs, df['test_loss'], 'r--', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Total Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Component losses
    loss_components = ['loss_ce', 'loss_bbox', 'loss_giou']
    for comp in loss_components:
        train_key = f'train_{comp}'
        if train_key in df.columns:
            axes[1].plot(epochs, df[train_key], label=f'Train {comp}', linewidth=1.5)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Loss Components (Training)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'loss_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. AP Metrics
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # mAP50
    if 'test_mAP50' in df.columns:
        axes[0].plot(epochs, df['test_mAP50'], 'g-', label='mAP50', linewidth=2, marker='o', markersize=4)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('AP')
    axes[0].set_title('mAP50')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 1])
    
    # Per-class AP
    ap_cols = [c for c in df.columns if c.startswith('test_AP_')]
    colors = ['#2ecc71', '#e74c3c', '#3498db', '#9b59b6']
    for i, col in enumerate(ap_cols):
        label = col.replace('test_AP_', 'AP_')
        axes[1].plot(epochs, df[col], label=label, linewidth=2, 
                    marker='s', markersize=4, color=colors[i % len(colors)])
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('AP')
    axes[1].set_title('Per-Class AP')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'ap_metrics.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. Recall Metrics
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Per-class Recall
    recall_cols = [c for c in df.columns if c.startswith('test_Recall_')]
    for i, col in enumerate(recall_cols):
        label = col.replace('test_Recall_', 'Recall_')
        axes[0].plot(epochs, df[col], label=label, linewidth=2, 
                    marker='^', markersize=4, color=colors[i % len(colors)])
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Recall')
    axes[0].set_title('Per-Class Recall')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 1])
    
    # AR@100 from COCO eval (index 8)
    if 'test_coco_eval_bbox' in df.columns:
        ar_100 = [row[8] if isinstance(row, list) and len(row) > 8 else 0 
                  for row in df['test_coco_eval_bbox']]
        axes[1].plot(epochs, ar_100, 'b-', label='AR@100', linewidth=2, marker='o', markersize=4)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Average Recall')
    axes[1].set_title('AR@100 (COCO)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'recall_metrics.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 4. Timing
    if 'train_time' in df.columns and 'val_time' in df.columns:
        fig, ax = plt.subplots(figsize=(10, 5))
        
        width = 0.35
        x = np.arange(len(epochs))
        
        ax.bar(x - width/2, df['train_time'], width, label='Train Time', color='#3498db', alpha=0.8)
        ax.bar(x + width/2, df['val_time'], width, label='Val Time', color='#e74c3c', alpha=0.8)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Time (seconds)')
        ax.set_title('Training and Validation Time per Epoch')
        ax.set_xticks(x)
        ax.set_xticklabels(epochs)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'timing.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"Generated plots: loss_curves.png, ap_metrics.png, recall_metrics.png, timing.png")
