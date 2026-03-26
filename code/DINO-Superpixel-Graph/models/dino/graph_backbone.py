# ------------------------------------------------------------------------
# DINO-Superpixel-Graph
# GPU-based Graph Attention Network Backbone for Superpixel Object Detection
# Replaces handcrafted 136-dim features with learnable CNN + GAT pipeline
# All operations run on GPU — zero CPU transfers during forward pass
# ------------------------------------------------------------------------

"""
Graph Backbone Module (GPU-only, Multi-Scale).

Pipeline:
    Image [B, 3, H, W]
    → SuperpixelCNN → Feature Map [B, C, H, W]
    → scatter_mean(slic_map) → Node Features [B, K, C]
    → GraphConstructor → edge_index + edge_attr (5D → projected)
    → GATModule (3-layer GATConv) → Node Embeddings [B, K, 256]
    → Multi-scale concat + pad → Tokens for SLICTransformer

Edge features (5D raw → projected):
    1. Color diff:      L2 of mean RGB between adjacent nodes
    2. Feature diff:    L2 of CNN features between adjacent nodes
    3. Texture diff:    L2 of mean gradient magnitude between adjacent nodes
    4. Spatial distance: Euclidean between centroids
    5. Boundary strength: Mean gradient magnitude at boundary pixels
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
from torch_geometric.nn import GATConv
from util.misc import NestedTensor


# =============================================================================
# SuperpixelCNN — Lightweight per-pixel feature extractor
# =============================================================================
class SuperpixelCNN(nn.Module):
    """3-layer Conv→BN→ReLU, extracts per-pixel features from images.

    Args:
        in_channels: number of input channels (3 for RGB)
        out_channels: number of output feature channels
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 64):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.out_channels = out_channels
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, images: torch.Tensor, debug: bool = False) -> torch.Tensor:
        """
        Args:
            images: [B, 3, H, W] normalized images
            debug: if True, print tensor stats
        Returns:
            feature_map: [B, out_channels, H, W]
        """
        if debug:
            print(f"  [SuperpixelCNN] Input: {images.shape}, "
                  f"mean={images.mean():.4f}, std={images.std():.4f}")

        out = self.layers(images)

        if debug:
            print(f"  [SuperpixelCNN] Output: {out.shape}, "
                  f"mean={out.mean():.4f}, std={out.std():.4f}")
        return out


# =============================================================================
# GraphConstructor — Builds graph structure on GPU from SLIC map
# =============================================================================
class GraphConstructor(nn.Module):
    """Builds edge_index and edge_features from SLIC map entirely on GPU.

    Args:
        raw_edge_dim: dimension of raw edge features (5)
        edge_proj_dim: dimension after projection for GAT
    """

    def __init__(self, raw_edge_dim: int = 5, edge_proj_dim: int = 32):
        super().__init__()
        self.edge_proj = nn.Linear(raw_edge_dim, edge_proj_dim)
        nn.init.xavier_uniform_(self.edge_proj.weight)
        nn.init.zeros_(self.edge_proj.bias)

        # Sobel kernels for texture/boundary features
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                               dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                               dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def build_adjacency(self, slic_map: torch.Tensor, K: int) -> torch.Tensor:
        """Build region adjacency edge_index from SLIC map on GPU.

        Args:
            slic_map: [H, W] integer superpixel IDs (single image)
            K: maximum superpixel ID + 1
        Returns:
            edge_index: [2, E] undirected edges between adjacent superpixels
        """
        H, W = slic_map.shape
        device = slic_map.device

        # Check right neighbors
        left = slic_map[:, :-1].flatten()   # [H*(W-1)]
        right = slic_map[:, 1:].flatten()   # [H*(W-1)]

        # Check bottom neighbors
        top = slic_map[:-1, :].flatten()    # [(H-1)*W]
        bottom = slic_map[1:, :].flatten()  # [(H-1)*W]

        # Concatenate all neighbor pairs
        src = torch.cat([left, top])
        dst = torch.cat([right, bottom])

        # Filter: only keep pairs where src != dst (different superpixels)
        # and both are valid (>= 0 and < K)
        valid = (src != dst) & (src >= 0) & (src < K) & (dst >= 0) & (dst < K)
        src = src[valid]
        dst = dst[valid]

        # Create undirected edges (both directions)
        edges = torch.stack([
            torch.cat([src, dst]),
            torch.cat([dst, src])
        ], dim=0)  # [2, 2*E_raw]

        # Remove duplicate edges: encode as unique pair IDs
        pair_ids = edges[0] * K + edges[1]
        unique_ids, inverse = torch.unique(pair_ids, return_inverse=True)

        # Get first occurrence of each unique edge
        perm = torch.arange(edges.shape[1], device=device)
        first_occ = torch.zeros(unique_ids.shape[0], dtype=torch.long, device=device)
        # scatter with min to find first occurrence
        first_occ.scatter_(0, inverse, perm)

        edge_index = edges[:, first_occ]  # [2, E_unique]

        return edge_index

    def compute_centroids(self, slic_map: torch.Tensor, K: int) -> torch.Tensor:
        """Compute normalized centroids for each superpixel.

        Args:
            slic_map: [H, W] integer superpixel IDs
            K: max superpixels
        Returns:
            centroids: [K, 2] normalized (cy, cx) in [0, 1]
        """
        H, W = slic_map.shape
        device = slic_map.device

        y_coords = torch.linspace(0, 1, H, device=device).view(H, 1).expand(H, W)
        x_coords = torch.linspace(0, 1, W, device=device).view(1, W).expand(H, W)

        sp_flat = slic_map.flatten()  # [H*W]
        y_flat = y_coords.flatten()   # [H*W]
        x_flat = x_coords.flatten()   # [H*W]

        valid = (sp_flat >= 0) & (sp_flat < K)
        sp_safe = sp_flat.clone()
        sp_safe[~valid] = 0

        # scatter_mean manually: sum + count
        y_sum = torch.zeros(K, device=device).scatter_add_(0, sp_safe, y_flat * valid.float())
        x_sum = torch.zeros(K, device=device).scatter_add_(0, sp_safe, x_flat * valid.float())
        counts = torch.zeros(K, device=device).scatter_add_(0, sp_safe, valid.float())
        safe_counts = counts.clamp(min=1)

        centroids = torch.stack([y_sum / safe_counts, x_sum / safe_counts], dim=1)
        return centroids  # [K, 2]

    def compute_edge_features(
        self,
        node_features: torch.Tensor,
        centroids: torch.Tensor,
        image: torch.Tensor,
        slic_map: torch.Tensor,
        edge_index: torch.Tensor,
        K: int,
    ) -> torch.Tensor:
        """Compute 5D raw edge features, then project.

        Args:
            node_features: [K, C] CNN features per superpixel
            centroids: [K, 2]
            image: [3, H, W] single image (normalized)
            slic_map: [H, W]
            edge_index: [2, E]
            K: max superpixels
        Returns:
            edge_attr: [E, edge_proj_dim] projected edge features
        """
        device = node_features.device
        src, dst = edge_index[0], edge_index[1]

        # 1. Color diff: L2 of mean RGB per superpixel
        # Denormalize image to [0,1]
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)
        img_01 = (image * std + mean).clamp(0, 1)

        # Compute mean RGB per superpixel
        H, W = slic_map.shape
        sp_flat = slic_map.flatten()
        valid = (sp_flat >= 0) & (sp_flat < K)
        sp_safe = sp_flat.clone()
        sp_safe[~valid] = 0

        rgb_flat = img_01.reshape(3, -1).t()  # [H*W, 3]
        rgb_sum = torch.zeros(K, 3, device=device)
        counts = torch.zeros(K, 1, device=device)
        for c in range(3):
            rgb_sum[:, c].scatter_add_(0, sp_safe, rgb_flat[:, c] * valid.float())
        counts[:, 0].scatter_add_(0, sp_safe, valid.float())
        rgb_mean = rgb_sum / counts.clamp(min=1)

        color_diff = (rgb_mean[src] - rgb_mean[dst]).norm(dim=1, keepdim=True)  # [E, 1]

        # 2. Feature diff: L2 of CNN node features
        feat_diff = (node_features[src] - node_features[dst]).norm(dim=1, keepdim=True)  # [E, 1]

        # 3. Texture diff: L2 of mean gradient magnitude per superpixel
        gray = img_01.mean(dim=0, keepdim=True).unsqueeze(0)  # [1, 1, H, W]
        grad_x = F.conv2d(gray, self.sobel_x, padding=1)
        grad_y = F.conv2d(gray, self.sobel_y, padding=1)
        grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8).squeeze()  # [H, W]

        grad_flat = grad_mag.flatten()
        grad_sum = torch.zeros(K, device=device).scatter_add_(0, sp_safe, grad_flat * valid.float())
        grad_mean = grad_sum / counts.squeeze().clamp(min=1)

        texture_diff = (grad_mean[src] - grad_mean[dst]).abs().unsqueeze(1)  # [E, 1]

        # 4. Spatial distance: Euclidean between centroids
        spatial_dist = (centroids[src] - centroids[dst]).norm(dim=1, keepdim=True)  # [E, 1]

        # 5. Boundary strength: mean gradient magnitude at boundary pixels
        # Find boundary pixels for each edge (pixels where src_sp and dst_sp are neighbors)
        # Approximate: average of gradient mean at both superpixels
        boundary_strength = ((grad_mean[src] + grad_mean[dst]) / 2).unsqueeze(1)  # [E, 1]

        # Concatenate and project
        raw_feats = torch.cat([
            color_diff,        # 1D
            feat_diff,         # 1D
            texture_diff,      # 1D
            spatial_dist,      # 1D
            boundary_strength, # 1D
        ], dim=1)  # [E, 5]

        edge_attr = self.edge_proj(raw_feats)  # [E, edge_proj_dim]
        return edge_attr

    def forward(
        self,
        slic_map: torch.Tensor,
        image: torch.Tensor,
        node_features: torch.Tensor,
        K: int,
        debug: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build graph for a single image.

        Args:
            slic_map: [H, W]
            image: [3, H, W]
            node_features: [K, C]
            K: max superpixels
            debug: print stats
        Returns:
            edge_index: [2, E]
            edge_attr: [E, edge_proj_dim]
            centroids: [K, 2]
        """
        centroids = self.compute_centroids(slic_map, K)
        edge_index = self.build_adjacency(slic_map, K)

        if edge_index.shape[1] == 0:
            # No edges — add self-loop to avoid empty graph
            edge_index = torch.zeros(2, 1, dtype=torch.long, device=slic_map.device)

        edge_attr = self.compute_edge_features(
            node_features, centroids, image, slic_map, edge_index, K
        )

        if debug:
            print(f"  [GraphConstructor] K={K}, edges={edge_index.shape[1]}, "
                  f"centroids={centroids.shape}, edge_attr={edge_attr.shape}")
            print(f"  [GraphConstructor] edge_attr mean={edge_attr.mean():.4f}, "
                  f"std={edge_attr.std():.4f}")

        return edge_index, edge_attr, centroids


# =============================================================================
# GATModule — 3-layer Graph Attention Network with edge features
# =============================================================================
class GATModule(nn.Module):
    """Multi-layer GATConv with edge features, residual connections, and LayerNorm.

    Args:
        input_dim: input node feature dimension
        hidden_dim: hidden dimension (per head, so actual = hidden_dim * heads for concat layers)
        output_dim: final output dimension (256 for DINO)
        num_layers: number of GAT layers
        edge_dim: edge feature dimension (from GraphConstructor projection)
        heads: number of attention heads
        dropout: dropout rate
    """

    def __init__(
        self,
        input_dim: int = 64,
        hidden_dim: int = 128,
        output_dim: int = 256,
        num_layers: int = 3,
        edge_dim: int = 32,
        heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.output_dim = output_dim

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)

        # GAT layers
        self.gat_layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(num_layers):
            if i == 0:
                in_ch = hidden_dim
            else:
                in_ch = hidden_dim  # after concat → projection, always hidden_dim

            if i < num_layers - 1:
                # Intermediate layers: concat heads, then project back
                out_ch = hidden_dim // heads  # per-head output
                concat = True
                # After concat: out_ch * heads = hidden_dim
            else:
                # Last layer: average heads, output final dim
                out_ch = output_dim
                concat = False

            self.gat_layers.append(
                GATConv(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    heads=heads if i < num_layers - 1 else 1,
                    concat=concat,
                    edge_dim=edge_dim,
                    dropout=dropout,
                    add_self_loops=True,
                    residual=False,  # we handle residual manually
                )
            )
            out_dim = out_ch * heads if concat else out_ch
            self.norms.append(nn.LayerNorm(out_dim))

        # Residual projection if dimensions change
        self.residual_proj = nn.Linear(hidden_dim, output_dim) if hidden_dim != output_dim else nn.Identity()
        self.output_norm = nn.LayerNorm(output_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        debug: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x: [N, input_dim] node features
            edge_index: [2, E]
            edge_attr: [E, edge_dim]
            debug: print tensor stats
        Returns:
            x: [N, output_dim] node embeddings
        """
        if debug:
            print(f"  [GATModule] Input: x={x.shape}, edges={edge_index.shape[1]}, "
                  f"mean={x.mean():.4f}, std={x.std():.4f}")

        # Input projection
        x = self.input_proj(x)
        x = self.input_norm(x)
        residual = x

        # GAT layers
        for i, (gat, norm) in enumerate(zip(self.gat_layers, self.norms)):
            x = gat(x, edge_index, edge_attr=edge_attr)
            x = F.elu(x)
            x = norm(x)

            if debug:
                print(f"  [GATModule] Layer {i}: {x.shape}, "
                      f"mean={x.mean():.4f}, std={x.std():.4f}")

        # Residual connection
        residual = self.residual_proj(residual)
        x = x + residual
        x = self.output_norm(x)

        if debug:
            print(f"  [GATModule] Output: {x.shape}, "
                  f"mean={x.mean():.4f}, std={x.std():.4f}")

        return x


# =============================================================================
# GraphFeatureExtractor — Top-level backbone (replaces SLICFeatureExtractor)
# =============================================================================
class GraphFeatureExtractor(nn.Module):
    """Graph backbone for DINO: CNN + GAT on superpixel graphs (multi-scale, GPU).

    Replaces SLICFeatureExtractor. Produces the exact same output format:
        tokens:       [bs, N_total, output_dim]
        centroids:    [bs, N_total, 2]
        padding_mask: [bs, N_total] — True = padded
        level_counts: [num_levels]

    Args:
        n_segments_per_level: list of SLIC n_segments for each scale
        output_dim: output feature dimension (256 for DINO)
        cnn_out_channels: CNN feature map channels
        gcn_hidden_dim: GATModule hidden dimension
        gcn_num_layers: number of GAT layers
        gcn_edge_dim: projected edge feature dimension
        gcn_heads: number of GAT attention heads
        max_superpixels_per_level: max superpixels per level for padding
        compactness: SLIC compactness (CPU fallback only)
        sigma: SLIC sigma (CPU fallback only)
        debug: print debug info
    """

    def __init__(
        self,
        n_segments_per_level: List[int] = [400, 200, 100],
        output_dim: int = 256,
        cnn_out_channels: int = 64,
        gcn_hidden_dim: int = 128,
        gcn_num_layers: int = 3,
        gcn_edge_dim: int = 32,
        gcn_heads: int = 4,
        max_superpixels_per_level: int = 500,
        compactness: float = 10.0,
        sigma: float = 1.0,
        debug: bool = False,
    ):
        super().__init__()
        self.n_segments_per_level = n_segments_per_level
        self.output_dim = output_dim
        self.debug = debug
        self.num_levels = len(n_segments_per_level)
        self.max_superpixels_per_level = max_superpixels_per_level
        self.compactness = compactness
        self.sigma = sigma

        # Number of output channels per level (all same, for compat with DINO)
        self.num_channels = [output_dim] * self.num_levels

        # Components
        self.cnn = SuperpixelCNN(in_channels=3, out_channels=cnn_out_channels)
        self.graph_constructor = GraphConstructor(
            raw_edge_dim=5, edge_proj_dim=gcn_edge_dim
        )
        self.gat = GATModule(
            input_dim=cnn_out_channels,
            hidden_dim=gcn_hidden_dim,
            output_dim=output_dim,
            num_layers=gcn_num_layers,
            edge_dim=gcn_edge_dim,
            heads=gcn_heads,
        )

    @staticmethod
    def _resize_slic_map(sp_map: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
        """Resize slic_map to match image dimensions using nearest interpolation."""
        sp_h, sp_w = sp_map.shape
        if sp_h == target_h and sp_w == target_w:
            return sp_map
        return F.interpolate(
            sp_map.float().unsqueeze(0).unsqueeze(0),
            size=(target_h, target_w), mode='nearest'
        ).squeeze(0).squeeze(0).long()

    def _run_slic_cpu(self, images: torch.Tensor, n_segments: int) -> torch.Tensor:
        """Fallback: run SLIC on-the-fly on CPU (for inference without pre-computed maps)."""
        from skimage.segmentation import slic as skimage_slic
        B, _, H, W = images.shape
        device = images.device
        slic_maps = torch.full((B, H, W), -1, dtype=torch.long, device=device)
        for i in range(B):
            img_np = images[i].detach().cpu().permute(1, 2, 0).numpy()
            img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img_np = np.clip(img_np, 0, 1).astype(np.float32)
            segments = skimage_slic(
                img_np, n_segments=n_segments,
                compactness=self.compactness, sigma=self.sigma,
                start_label=0, channel_axis=2,
            )
            slic_maps[i] = torch.from_numpy(segments.astype(np.int64)).to(device)
        return slic_maps

    def _scatter_mean_features(
        self,
        feature_map: torch.Tensor,
        slic_map: torch.Tensor,
        K: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pool CNN features per superpixel via scatter_mean.

        Args:
            feature_map: [C, H, W] single image feature map
            slic_map: [H, W] superpixel IDs
            K: max superpixels
        Returns:
            node_features: [K, C]
            valid_mask: [K] True = valid superpixel
        """
        C = feature_map.shape[0]
        device = feature_map.device

        sp_flat = slic_map.flatten()  # [H*W]
        valid = (sp_flat >= 0) & (sp_flat < K)
        sp_safe = sp_flat.clone()
        sp_safe[~valid] = 0

        feat_flat = feature_map.reshape(C, -1).t()  # [H*W, C]

        # scatter_add for sum
        node_sum = torch.zeros(K, C, device=device)
        sp_idx = sp_safe.unsqueeze(1).expand(-1, C)  # [H*W, C]
        node_sum.scatter_add_(0, sp_idx, feat_flat * valid.unsqueeze(1).float())

        # counts
        counts = torch.zeros(K, device=device)
        counts.scatter_add_(0, sp_safe, valid.float())
        safe_counts = counts.clamp(min=1)

        node_features = node_sum / safe_counts.unsqueeze(1)  # [K, C]
        valid_mask = counts > 0

        return node_features, valid_mask

    def forward(self, tensor_list: NestedTensor, targets=None) -> dict:
        """Forward pass: CNN → scatter_mean → GAT → multi-scale token sequences.

        Args:
            tensor_list: NestedTensor with tensors [bs, 3, H, W] and mask [bs, H, W]
            targets: list of dicts with 'slic_maps': {n_seg: [H, W] tensor}
        Returns:
            dict with tokens, centroids, padding_mask, level_counts
        """
        images = tensor_list.tensors  # [B, 3, H, W]
        B, _, H, W = images.shape
        device = images.device

        if self.debug:
            print(f"[GraphFeatureExtractor] Input: images={images.shape}")

        # Step 1: CNN feature extraction (shared across all scales)
        feature_map = self.cnn(images, debug=self.debug)  # [B, C, H, W]

        # Step 2: Process each scale level
        level_features = []
        level_centroids = []
        level_masks = []
        level_counts = []

        for lvl, n_seg in enumerate(self.n_segments_per_level):
            K = self.max_superpixels_per_level

            # Get SLIC maps for this level
            has_precomputed = (
                targets is not None
                and len(targets) > 0
                and 'slic_maps' in targets[0]
                and n_seg in targets[0]['slic_maps']
            )

            if has_precomputed:
                slic_maps = torch.stack([
                    self._resize_slic_map(t['slic_maps'][n_seg].to(device), H, W)
                    for t in targets
                ])  # [B, H, W]
            else:
                slic_maps = self._run_slic_cpu(images, n_seg)

            # Per-sample: scatter_mean → graph construction → GAT
            batch_node_embs = []
            batch_centroids = []
            batch_valid_masks = []

            for b in range(B):
                # scatter_mean CNN features → node features
                node_feat, valid_mask = self._scatter_mean_features(
                    feature_map[b], slic_maps[b], K
                )  # [K, C], [K]

                # Build graph
                edge_index, edge_attr, centroids = self.graph_constructor(
                    slic_maps[b], images[b], node_feat, K,
                    debug=(self.debug and b == 0 and lvl == 0),
                )

                # GAT forward
                node_embs = self.gat(
                    node_feat, edge_index, edge_attr,
                    debug=(self.debug and b == 0 and lvl == 0),
                )  # [K, output_dim]

                batch_node_embs.append(node_embs)
                batch_centroids.append(centroids)
                batch_valid_masks.append(valid_mask)

            # Stack across batch
            features = torch.stack(batch_node_embs)     # [B, K, output_dim]
            centroids = torch.stack(batch_centroids)     # [B, K, 2]
            mask = torch.stack(batch_valid_masks)         # [B, K] True = valid

            # Find actual max valid count across batch
            valid_counts = mask.sum(dim=1)
            max_valid = int(valid_counts.max().item())
            max_valid = max(max_valid, 1)

            # Trim to actual size needed
            features = features[:, :max_valid]
            centroids = centroids[:, :max_valid]
            mask = mask[:, :max_valid]

            level_features.append(features)
            level_centroids.append(centroids)
            level_masks.append(mask)
            level_counts.append(max_valid)

        # Step 3: Concatenate all levels
        n_total = sum(level_counts)

        all_tokens = torch.zeros(B, n_total, self.output_dim, device=device)
        all_centroids = torch.zeros(B, n_total, 2, device=device)
        all_masks = torch.ones(B, n_total, dtype=torch.bool, device=device)  # True = padded

        offset = 0
        for lvl in range(self.num_levels):
            k = level_counts[lvl]
            all_tokens[:, offset:offset + k] = level_features[lvl]
            all_centroids[:, offset:offset + k] = level_centroids[lvl]
            # Convert valid mask (True=valid) to padding mask (True=padded)
            all_masks[:, offset:offset + k] = ~level_masks[lvl]
            offset += k

        if self.debug:
            print(f"[GraphFeatureExtractor] Output: tokens={all_tokens.shape}, "
                  f"centroids={all_centroids.shape}, mask={all_masks.shape}, "
                  f"level_counts={level_counts}")
            print(f"[GraphFeatureExtractor] tokens mean={all_tokens.mean():.4f}, "
                  f"std={all_tokens.std():.4f}")

        return {
            'tokens': all_tokens,        # [bs, N_total, output_dim]
            'centroids': all_centroids,   # [bs, N_total, 2]
            'padding_mask': all_masks,    # [bs, N_total] — True = padded
            'level_counts': level_counts, # list[int]
        }
