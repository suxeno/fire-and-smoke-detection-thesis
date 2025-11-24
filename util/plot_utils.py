"""
Plotting utilities to visualize training logs.
"""
import torch
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path, PurePath


def plot_logs(logs, fields=('loss', 'mAP', 'mAP50', 'Recall'), ewm_col=0, log_name='log.txt'):
    '''
    Function to plot specific fields from training log(s). Plots both training and validation results.
    Handles multiple validation categories (CV, UAV, RS).
    '''
    func_name = "plot_utils.py::plot_logs"

    if not isinstance(logs, list):
        if isinstance(logs, PurePath):
            logs = [logs]
        else:
            raise ValueError(f"{func_name} - invalid argument for logs parameter.")

    for i, dir in enumerate(logs):
        if not isinstance(dir, PurePath):
            raise ValueError(f"{func_name} - non-Path object in logs argument")
        if not dir.exists():
            raise ValueError(f"{func_name} - invalid directory: {dir}")
        fn = Path(dir / log_name)
        if not fn.exists():
            print(f"-> missing {log_name} in {dir}")
            return

    # Load log files
    dfs = [pd.read_json(Path(p) / log_name, lines=True) for p in logs]

    # Create subplots
    fig, axs = plt.subplots(ncols=len(fields), figsize=(5 * len(fields), 5))
    if len(fields) == 1:
        axs = [axs]

    # Define styles for different categories
    styles = {
        'train': {'color': 'blue', 'style': '-', 'label': 'Train'},
        'val': {'color': 'black', 'style': '--', 'label': 'Val (Avg)'},
        'val_CV': {'color': 'orange', 'style': '--', 'label': 'Val CV'},
        'val_UAV': {'color': 'green', 'style': '--', 'label': 'Val UAV'},
        'val_RS': {'color': 'red', 'style': '--', 'label': 'Val RS'},
    }

    for df in dfs:
        for j, field in enumerate(fields):
            ax = axs[j]
            
            # Find all columns matching the field
            # We look for: train_{field}, val_{field}, val_CV_{field}, etc.
            
            # 1. Train
            if f'train_{field}' in df.columns:
                data = df[f'train_{field}'].interpolate().ewm(com=ewm_col).mean()
                ax.plot(data, color=styles['train']['color'], linestyle=styles['train']['style'], label=styles['train']['label'])
            
            # 2. Val Average
            if f'val_{field}' in df.columns:
                data = df[f'val_{field}'].interpolate().ewm(com=ewm_col).mean()
                ax.plot(data, color=styles['val']['color'], linestyle=styles['val']['style'], label=styles['val']['label'])
            
            # 3. Val Categories
            for cat in ['CV', 'UAV', 'RS']:
                col_name = f'val_{cat}_{field}'
                if col_name in df.columns:
                    data = df[col_name].interpolate().ewm(com=ewm_col).mean()
                    style = styles.get(f'val_{cat}', {'color': 'gray', 'style': '--'})
                    ax.plot(data, color=style['color'], linestyle=style['style'], label=style['label'])
            
            # Special handling for mAP/AP if not found directly
            if field == 'mAP' and 'val_AP' not in df.columns:
                # Try to find val_AP or val_CV_AP etc if field is just 'mAP'
                # But usually we pass 'mAP' and expect 'val_mAP' or 'val_AP'
                # If the user passes 'mAP', we map it to 'AP' in the log
                if 'val_AP' in df.columns:
                     data = df['val_AP'].interpolate().ewm(com=ewm_col).mean()
                     ax.plot(data, color=styles['val']['color'], linestyle=styles['val']['style'], label=styles['val']['label'])

            ax.set_title(field)
            ax.set_xlabel('Epoch')
            ax.grid(True, linestyle=':', alpha=0.6)
            
            # Deduplicate legend labels
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys())

    plt.tight_layout()


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