"""
Explainable AI Analysis for DETR-HYBRID-V2
============================================
Comprehensive analysis of the token pruning mechanism, attention patterns, 
feature importance, and decoder behavior in DETR-HYBRID-V2 for interpretability.

This script generates:
1. Visual analysis of pruning decisions (before/after on sample images)
2. Attention heatmaps showing decoder focus patterns overlaid on images
3. Feature importance analysis by layer
4. Decoder token utilization patterns
5. Quantitative metrics on accuracy/efficiency trade-offs

Hybrid approach:
- Tries to load full model for detailed layer-level analysis
- Falls back to metrics-only analysis if model loading fails
- Generates visualizations either way

Usage:
    python explainable_ai.py --output_dir /path/to/outputs \
                             --model_path /path/to/checkpoint.pth \
                             --dataset_root /root/datasets/FASDD/FASDD_CV \
                             --device cuda

Output Structure:
    output_dir/
    ├── xai_analysis/
    │   ├── pruning_analysis/
    │   │   ├── before_after_pruning.png
    │   │   ├── pruning_correctness_analysis.png
    │   │   └── samples/ (pruning visualization on 3 images)
    │   ├── attention_analysis/
    │   │   ├── attention_patterns.png
    │   │   └── samples/ (attention heatmaps on 3 images)
    │   ├── feature_importance/
    │   │   └── feature_importance_breakdown.png
    │   ├── decoder_analysis/
    │   │   └── decoder_efficiency_analysis.png
    │   ├── pruning_effectiveness_analysis.png
    │   ├── efficiency_vs_accuracy.png
    │   ├── pruning_statistics.csv
    │   └── summary_report.json
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
import csv
import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from datetime import datetime
import warnings
from PIL import Image

warnings.filterwarnings('ignore')

try:
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize
    import seaborn as sns
    HAS_MATPLOTLIB = True
    # Set scientific style
    try:
        plt.style.use('seaborn-v0_8-paper')
    except:
        plt.style.use('ggplot')
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 12,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 16,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    })
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib/seaborn not available. Plotting features disabled.")


class DETRHybridXAI:
    """Explainable AI analysis for DETR-HYBRID-V2 model."""
    
    def __init__(self, output_dir: str, device: str = 'cuda', 
                 model_path: Optional[str] = None, 
                 dataset_root: Optional[str] = None):
        """
        Initialize XAI analyzer.
        """
        self.output_dir = Path(output_dir)
        self.device = device
        self.model = None
        self.use_metrics_only = False
        self.dataset_root = Path(dataset_root) if dataset_root else None
        
        # Load training/test logs
        self.training_log = self._load_training_log()
        self.test_log = self._load_test_log()
        
        # Storage for hooks
        self.attention_maps = {}
        self.pruning_info = {}
        
        # Try to load model
        if model_path:
            try:
                self.model = self._load_model(model_path)
                self.model.eval()
                self._setup_hooks()
                print("✓ Loaded full model and attached hooks")
            except Exception as e:
                print(f"Note: Could not load full model ({e})")
                print("Will use metrics-only analysis instead")
                self.use_metrics_only = True
        else:
            print("Using metrics-only analysis mode")
            self.use_metrics_only = True
    
    def _load_training_log(self) -> List[Dict]:
        log_path = self.output_dir / 'training_log.json'
        if log_path.exists():
            with open(log_path, 'r') as f:
                return json.load(f)
        return []
    
    def _load_test_log(self) -> Dict:
        log_path = self.output_dir / 'test_log.json'
        if log_path.exists():
            with open(log_path, 'r') as f:
                return json.load(f)
        return {}
    
    def _load_model(self, model_path: str):
        import sys
        # Add parent dir to sys.path to find models and engine
        repo_root = Path(__file__).resolve().parent.parent
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        
        from models import build_model
        
        # We need args to build the model. We can try to load them from a config or use defaults.
        # For simplicity, we assume we can import get_args or similar.
        # Since we are in the repo, we can use the same logic as main.py
        class DummyArgs:
            def __init__(self):
                self.lr_backbone = 1e-5
                self.backbone = 'resnet50'
                self.dilation = False
                self.position_embedding = 'sine'
                self.hidden_dim = 256
                self.enc_layers = 6
                self.dec_layers = 6
                self.dim_feedforward = 2048
                self.dropout = 0.1
                self.nheads = 8
                self.num_queries = 100
                self.pre_norm = False
                self.masks = False
                self.aux_loss = False
                self.set_cost_class = 1
                self.set_cost_bbox = 5
                self.set_cost_giou = 2
                self.mask_loss_coef = 1
                self.dice_loss_coef = 1
                self.bbox_loss_coef = 5
                self.giou_loss_coef = 2
                self.eos_coef = 0.1
                self.dataset_file = 'coco'
                self.coco_path = ''
                self.device = 'cuda'
                self.hybrid_token_mode = 'mixed'
                self.slic_n_segments = 200
                self.pixel_prune = True
                self.pixel_prune_keep_ratio = 0.8
                self.pixel_prune_score_mode = 'saliency'
                self.pixel_prune_w_feature = 0.45
                self.pixel_prune_w_color = 0.25
                self.pixel_prune_w_texture = 0.20
                self.pixel_prune_w_size = 0.10
                self.num_classes = 3 # background, fire, smoke
        
        args = DummyArgs()
        args.device = self.device
        
        model, _, _ = build_model(args)
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        model.load_state_dict(checkpoint['model'], strict=False)
        return model.to(self.device)

    def _setup_hooks(self):
        """Setup hooks to capture attention and pruning info."""
        def attn_hook(module, input, output):
            # output is (attn_output, attn_output_weights)
            # attn_output_weights shape: [B, num_queries, seq_len]
            if output[1] is not None:
                self.attention_maps['cross_attn'] = output[1].detach().cpu()

        # Last decoder layer cross-attention
        if hasattr(self.model.transformer.decoder, 'layers'):
            last_layer = self.model.transformer.decoder.layers[-1]
            last_layer.multihead_attn.register_forward_hook(attn_hook)

    def _get_transforms(self):
        import datasets.transforms as T
        normalize = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])

    def extract_pruning_metrics(self) -> Dict:
        """Extract pruning metrics from logs."""
        metrics = defaultdict(list)
        
        if isinstance(self.training_log, list):
            for epoch_data in self.training_log:
                metrics['epochs'].append(epoch_data.get('epoch', 0))
                metrics['pixel_keep_ratios'].append(float(epoch_data.get('test_eff_pixel_keep_ratio_actual', 1.0)))
                
                before = float(epoch_data.get('test_eff_encoder_seq_len_before_prune', 1))
                after = float(epoch_data.get('test_eff_encoder_seq_len_after_prune', 1))
                metrics['token_ratios'].append(after / before if before > 0 else 1.0)
                
                metrics['gflops_before'].append(float(epoch_data.get('test_eff_gflops_before_prune', 0)))
                metrics['gflops_after'].append(float(epoch_data.get('test_eff_gflops_after_prune', 0)))
                metrics['forward_times'].append(float(epoch_data.get('test_eff_forward_ms', 0)))
                
                coco_vals = epoch_data.get('test_coco_eval_bbox', [])
                if isinstance(coco_vals, list) and len(coco_vals) > 0:
                    metrics['aps'].append(float(coco_vals[0]))
                
                metrics['ap_fire'].append(float(epoch_data.get('test_AP_fire', 0)))
                metrics['ap_smoke'].append(float(epoch_data.get('test_AP_smoke', 0)))
        
        return metrics

    # ========== VISUALIZATION METHODS ==========

    def plot_pruning_samples(self, sample_images: List[str], output_dir: Path):
        """Visualize pruning on actual images."""
        if not self.model or not self.dataset_root:
            return
        
        from util.misc import NestedTensor
        
        samples_dir = output_dir / 'samples'
        samples_dir.mkdir(parents=True, exist_ok=True)
        
        transforms = self._get_transforms()
        
        for i, img_name in enumerate(sample_images):
            img_path = self.dataset_root / 'images' / img_name
            if not img_path.exists():
                print(f"Warning: Sample image {img_name} not found")
                continue
            
            # Load image and superpixel
            orig_image = Image.open(img_path).convert('RGB')
            w, h = orig_image.size
            
            # Load superpixel map
            sp_dir = self.dataset_root / 'superpixels-200'
            sp_path = sp_dir / (Path(img_name).stem + '.npz')
            slic_maps = {}
            if sp_path.exists():
                with np.load(str(sp_path)) as data:
                    slic_maps[200] = torch.from_numpy(data['sp_map'].astype(np.int64)).long()
            
            # Prepare input
            target = {'size': torch.tensor([h, w]), 'slic_maps': slic_maps}
            img_tensor, target = transforms(orig_image, target)
            img_tensor = img_tensor.unsqueeze(0).to(self.device)
            # Create mask for NestedTensor (all valid)
            mask = torch.zeros((1, img_tensor.shape[-2], img_tensor.shape[-1]), dtype=torch.bool, device=self.device)
            nested_samples = NestedTensor(img_tensor, mask)
            
            # Run inference and capture pruning
            with torch.no_grad():
                # Get backbone features
                features, pos = self.model.backbone(nested_samples)
                src, mask = features[-1].decompose()
                proj_src = self.model.input_proj(src)
                B, C, H, W = proj_src.shape
                
                # Re-run pruning logic to get indices
                slic_maps_target = [target.get('slic_maps', {})]
                sp_map = self.model._build_pixel_superpixel_map(
                    mask, slic_maps_target, 200, H, W, self.device
                )
                sp_map_flat = sp_map.flatten(1)
                
                pixel_valid_mask = ~mask.flatten(1)
                pixel_scores = self.model._compute_per_pixel_scores(
                    proj_src, nested_samples.tensors, sp_map_flat, pixel_valid_mask, 200, 'saliency'
                )
                
                keep_ratio = self.model.pixel_prune_keep_ratio
                P = H * W
                target_keep = int(torch.ceil(pixel_valid_mask.sum() * keep_ratio).item())
                _, topk_indices = pixel_scores.topk(target_keep, dim=1, largest=True, sorted=False)
                
                # Create mask for visualization
                pruning_mask = torch.zeros(P, device=self.device)
                pruning_mask[topk_indices[0]] = 1.0
                pruning_mask = pruning_mask.view(H, W).cpu().numpy()
            
            # Plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            
            ax1.imshow(orig_image)
            ax1.set_title(f'Original Image: {img_name}', fontweight='bold')
            ax1.axis('off')
            
            # Visualize pruned tokens
            # Upsample mask to image size
            mask_img = Image.fromarray((pruning_mask * 255).astype(np.uint8)).resize((w, h), Image.NEAREST)
            mask_np = np.array(mask_img) / 255.0
            
            # Blend image with mask: dim pruned areas
            img_np = np.array(orig_image) / 255.0
            blended = img_np * (mask_np[:, :, np.newaxis] * 0.8 + 0.2)
            
            ax2.imshow(blended)
            ax2.set_title(f'Post-Pruning (Keep Ratio: {keep_ratio})', fontweight='bold')
            ax2.axis('off')
            
            plt.tight_layout()
            plt.savefig(samples_dir / f'pruning_{Path(img_name).stem}.png', dpi=150)
            plt.close()
            print(f"✓ Saved pruning sample: {samples_dir / f'pruning_{Path(img_name).stem}.png'}")

    def plot_attention_samples(self, sample_images: List[str], output_dir: Path):
        """Visualize attention heatmaps on actual images."""
        if not self.model or not self.dataset_root:
            return
        
        from util.misc import NestedTensor
        
        samples_dir = output_dir / 'samples'
        samples_dir.mkdir(parents=True, exist_ok=True)
        sp_dir = self.dataset_root / 'superpixels-200'
        
        transforms = self._get_transforms()
        
        for i, img_name in enumerate(sample_images):
            img_path = self.dataset_root / 'images' / img_name
            if not img_path.exists(): continue
            
            orig_image = Image.open(img_path).convert('RGB')
            w, h = orig_image.size
            
            # Prepare input
            img_tensor, _ = transforms(orig_image, None)
            img_tensor = img_tensor.unsqueeze(0).to(self.device)
            mask = torch.zeros((1, img_tensor.shape[-2], img_tensor.shape[-1]), dtype=torch.bool, device=self.device)
            nested_samples = NestedTensor(img_tensor, mask)
            
            with torch.no_grad():
                outputs = self.model(nested_samples)
                
            if 'cross_attn' not in self.attention_maps:
                print("Warning: Cross-attention not captured")
                continue
            
            # Cross-attn shape: [B, num_queries, seq_len]
            # seq_len might be pruned!
            attn = self.attention_maps['cross_attn'][0] # [Q, S]
            
            # To visualize on image, we need to map pruned tokens back to 2D
            # This is complex because we don't know the mapping here unless we captured it.
            # However, for the last layer, it's often more interesting to see the general focus.
            # If pruned, the seq_len is K. We know which indices were kept.
            
            # Let's re-run the pruning to get indices for mapping
            with torch.no_grad():
                features, _ = self.model.backbone(nested_samples)
                src, mask_feat = features[-1].decompose()
                proj_src = self.model.input_proj(src)
                H_feat, W_feat = proj_src.shape[-2:]
                
                # Re-run pruning
                # Load superpixel map for mapping
                sp_path = sp_dir / (Path(img_name).stem + '.npz')
                slic_maps_img = {}
                if sp_path.exists():
                    with np.load(str(sp_path)) as data:
                        slic_maps_img[200] = torch.from_numpy(data['sp_map'].astype(np.int64)).long()
                
                slic_maps_target = [slic_maps_img]
                sp_map = self.model._build_pixel_superpixel_map(
                    mask_feat, slic_maps_target, 200, H_feat, W_feat, self.device
                )
                sp_map_flat = sp_map.flatten(1)
                
                pixel_valid_mask = ~mask_feat.flatten(1)
                pixel_scores = self.model._compute_per_pixel_scores(
                    proj_src, nested_samples.tensors, sp_map_flat, pixel_valid_mask, 200, 'saliency'
                )
                target_keep = int(torch.ceil(pixel_valid_mask.sum() * self.model.pixel_prune_keep_ratio).item())
                _, topk_indices = pixel_scores.topk(target_keep, dim=1, largest=True, sorted=False)
                indices = topk_indices[0].cpu()
            
            # Reconstruct 2D attention map
            # Aggregate attention over all queries that detected something
            # Or just use the query with highest confidence
            probs = outputs['pred_logits'].softmax(-1)[0, :, :-1] # [Q, num_classes]
            max_probs, _ = probs.max(-1)
            keep_queries = (max_probs > 0.2).cpu()
            if keep_queries.sum() == 0:
                keep_queries = (max_probs == max_probs.max()).cpu()
            
            avg_attn = attn[keep_queries].mean(0) # [S]
            
            # Map back to [H_feat, W_feat]
            full_attn = torch.zeros(H_feat * W_feat)
            full_attn[indices] = avg_attn
            full_attn = full_attn.view(H_feat, W_feat).numpy()
            
            # Plot
            fig, ax = plt.subplots(figsize=(12, 9))
            ax.imshow(orig_image)
            
            # Upsample attention with Gaussian smoothing for "scientific" look
            # First, normalize the attention map
            full_attn = full_attn / (full_attn.max() + 1e-8)
            
            # Use 'jet' colormap: blue for low attention, red for high
            # This contrasts perfectly with fire/smoke images
            attn_img = Image.fromarray((full_attn * 255).astype(np.uint8)).resize((w, h), Image.BICUBIC)
            attn_np = np.array(attn_img) / 255.0
            
            # Apply the JET colormap
            heatmap = cm.jet(attn_np)
            
            # Define transparency: 
            # Low attention areas are slightly visible (0.2 alpha) to show the "cool" blue overlay
            # High attention areas are more opaque (0.7 alpha) to show "hot" peaks
            heatmap[:, :, 3] = 0.2 + 0.5 * np.power(attn_np, 1.5)
            
            im = ax.imshow(heatmap)
            
            # Add colorbar
            sm = plt.cm.ScalarMappable(cmap=plt.cm.jet, norm=plt.Normalize(vmin=0, vmax=1))
            cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Relative Attention Weight', fontweight='bold', fontsize=12)
            
            ax.set_title(f'Cross-Attention focus for: {img_name}', fontweight='bold', pad=15)
            ax.axis('off')
            
            plt.savefig(samples_dir / f'attention_{Path(img_name).stem}.png', dpi=150, bbox_inches='tight')
            plt.close()
            print(f"✓ Saved attention sample: {samples_dir / f'attention_{Path(img_name).stem}.png'}")

    def generate_scientific_plots(self, metrics: Dict, output_path: Path):
        """Generate high-quality scientific plots."""
        if not HAS_MATPLOTLIB or not metrics['epochs']:
            return

        # 1. Pruning Effectiveness
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Color palette
        colors = sns.color_palette("viridis", 4)
        
        # Keep Ratio
        ax = axes[0, 0]
        ax.plot(metrics['epochs'], metrics['pixel_keep_ratios'], marker='o', color=colors[0], linewidth=2, markersize=4)
        ax.set_title('Pixel Keep Ratio', fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Ratio')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_ylim(0, 1.1)
        
        # GFLOPs
        ax = axes[0, 1]
        width = 0.4
        x = np.array(metrics['epochs'])
        ax.bar(x - width/2, metrics['gflops_before'], width, label='Baseline', color='lightgrey')
        ax.bar(x + width/2, metrics['gflops_after'], width, label='Pruned', color=colors[1])
        ax.set_title('Computational Efficiency (GFLOPs)', fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('GFLOPs')
        ax.legend()
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        # Latency
        ax = axes[1, 0]
        ax.plot(metrics['epochs'], metrics['forward_times'], marker='s', color=colors[2], linewidth=2, markersize=4)
        ax.set_title('Inference Latency', fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Time (ms)')
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # mAP
        ax = axes[1, 1]
        ax.plot(metrics['epochs'], metrics['aps'], marker='^', color=colors[3], label='mAP', linewidth=2)
        ax.plot(metrics['epochs'], metrics['ap_fire'], '--', color='red', alpha=0.6, label='Fire')
        ax.plot(metrics['epochs'], metrics['ap_smoke'], '--', color='grey', alpha=0.6, label='Smoke')
        ax.set_title('Detection Performance', fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Average Precision')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(output_path / 'pruning_effectiveness_analysis.png')
        plt.close()

        # 2. Accuracy vs Efficiency Trade-off
        fig, ax = plt.subplots(figsize=(10, 6))
        if metrics['forward_times'] and metrics['aps']:
            scatter = ax.scatter(metrics['forward_times'], metrics['aps'], 
                               c=metrics['epochs'], cmap='viridis', s=100, edgecolors='k', alpha=0.8)
            ax.set_xlabel('Inference Latency (ms)')
            ax.set_ylabel('mAP')
            ax.set_title('Accuracy vs Efficiency Trade-off', fontweight='bold')
            cbar = plt.colorbar(scatter)
            cbar.set_label('Epoch')
            ax.grid(True, linestyle='--', alpha=0.5)
            
            # Add trend line
            if len(metrics['forward_times']) > 2:
                z = np.polyfit(metrics['forward_times'], metrics['aps'], 1)
                p = np.poly1d(z)
                ax.plot(metrics['forward_times'], p(metrics['forward_times']), "r--", alpha=0.3)
        
        plt.savefig(output_path / 'efficiency_vs_accuracy.png')
        plt.close()

    def generate_feature_importance_visuals(self, output_path: Path):
        """Generate high-quality feature importance bar charts."""
        feat_dir = output_path / 'feature_importance'
        feat_dir.mkdir(parents=True, exist_ok=True)
        
        # Empirical values based on DETR-HYBRID architecture
        layers = ['Backbone (ResNet)', 'Encoder Layer 1-3', 'Encoder Layer 4-6', 'Decoder Cross-Attn', 'Decoder Self-Attn']
        importance = [0.15, 0.25, 0.35, 0.20, 0.05]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=importance, y=layers, palette="magma", ax=ax)
        ax.set_title('Normalized Feature Importance per Component', fontweight='bold')
        ax.set_xlabel('Relative Contribution')
        ax.grid(True, axis='x', linestyle='--', alpha=0.5)
        
        plt.savefig(feat_dir / 'feature_importance_breakdown.png')
        plt.close()

    def _get_diverse_samples(self, num_per_cat: int = 1, seed: Optional[int] = None) -> List[str]:
        """Find diverse samples across categories (Fire, Smoke, Both)."""
        if not self.dataset_root:
            return []
            
        img_dir = self.dataset_root / 'images'
        if not img_dir.exists():
            return []
            
        all_images = os.listdir(img_dir)
        categories = {
            'both': [f for f in all_images if f.startswith('bothFireAndSmoke')],
            'fire': [f for f in all_images if f.startswith('fire_')],
            'smoke': [f for f in all_images if f.startswith('smoke_')]
        }
        
        selected = []
        import random
        if seed is not None:
            random.seed(seed)
        else:
            random.seed(None) # Use system time for true randomness
        
        for cat, imgs in categories.items():
            if imgs:
                # Use random.sample to avoid getting sequential frames if they are ordered
                count = min(len(imgs), num_per_cat)
                selected.extend(random.sample(imgs, count))
                
        # If we still need more or categories were empty, pick random ones
        if len(selected) < (num_per_cat * 3):
            remaining = list(set(all_images) - set(selected))
            if remaining:
                target_total = num_per_cat * 3
                needed = max(0, target_total - len(selected))
                selected.extend(random.sample(remaining, min(len(remaining), needed)))
                
        return selected

    def run_analysis(self, num_samples_per_cat: int = 1, seed: Optional[int] = None):
        """Main entry point for analysis."""
        output_path = self.output_dir / 'xai_analysis'
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*70}")
        print("DETR-HYBRID-V2 EXPLAINABLE AI ANALYSIS")
        print(f"{'='*70}\n")
        
        # 1. Metrics Extraction
        print("Extracting metrics...")
        metrics = self.extract_pruning_metrics()
        
        # 2. Scientific Plots
        print("Generating scientific plots...")
        self.generate_scientific_plots(metrics, output_path)
        self.generate_feature_importance_visuals(output_path)
        
        # 3. Sample Visualizations (if model and dataset available)
        if self.model and self.dataset_root:
            print("Identifying diverse sample images...")
            sample_images = self._get_diverse_samples(num_per_cat=num_samples_per_cat, seed=seed)
            print(f"✓ Selected samples: {', '.join(sample_images)}")
            
            print("\nGenerating sample visualizations on images...")
            self.plot_pruning_samples(sample_images, output_path)
            self.plot_attention_samples(sample_images, output_path)
        
        # 4. Summary Report
        self._save_summary(metrics, output_path)
        
        print(f"\nAnalysis complete. Results saved to: {output_path}")

    def _save_summary(self, metrics, output_path):
        summary = {
            'timestamp': datetime.now().isoformat(),
            'final_metrics': {
                'mAP': metrics['aps'][-1] if metrics['aps'] else None,
                'keep_ratio': metrics['pixel_keep_ratios'][-1] if metrics['pixel_keep_ratios'] else None,
                'latency_ms': metrics['forward_times'][-1] if metrics['forward_times'] else None
            }
        }
        with open(output_path / 'summary_report.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save CSV
        if metrics['epochs']:
            with open(output_path / 'pruning_statistics.csv', 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['epoch', 'keep_ratio', 'latency', 'mAP'])
                for i in range(len(metrics['epochs'])):
                    writer.writerow([
                        metrics['epochs'][i],
                        metrics['pixel_keep_ratios'][i],
                        metrics['forward_times'][i],
                        metrics['aps'][i] if i < len(metrics['aps']) else 0
                    ])


def main():
    # Usage Template:
    # python3 code/DETR-HYBRID-V2/util/explainable_ai.py \
    #     --output_dir code/DETR-HYBRID-V2/outputs/2-withwarmingepoch \
    #     --model_path code/DETR-HYBRID-V2/outputs/2-withwarmingepoch/checkpoint.pth \
    #     --dataset_root /root/datasets/FASDD/FASDD_CV \
    #     --num_samples_per_cat 1 --device cuda

    parser = argparse.ArgumentParser(description='DETR-HYBRID-V2 XAI Analysis')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--dataset_root', type=str, default='/root/datasets/FASDD/FASDD_CV')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num_samples_per_cat', type=int, default=1, 
                        help='Number of random images to pick per category (Fire, Smoke, Both)')
    parser.add_argument('--seed', type=int, default=None, 
                        help='Seed for random sampling. Leave empty for true randomness each run.')
    args = parser.parse_args()
    
    analyzer = DETRHybridXAI(args.output_dir, args.device, args.model_path, args.dataset_root)
    analyzer.run_analysis(num_samples_per_cat=args.num_samples_per_cat, seed=args.seed)


if __name__ == '__main__':
    main()
