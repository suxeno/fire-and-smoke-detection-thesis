# ------------------------------------------------------------------------
# DINO-SLIC V2
# Learnable Superpixel Backbone (Multi-Scale)
# Replaces handcrafted 136D features with learnable pixel-level embedding
# + superpixel-aware aggregation (SuIT-style).
# All operations run on GPU — fully differentiable end-to-end.
# ------------------------------------------------------------------------

"""
Learnable SLIC Superpixel Backbone Module (GPU-only, Multi-Scale).

Pipeline (per scale level):
    Image [B, 3, H, W]
    → Conv2d(3→base_dim, 7×7, stride=2) + BN + GELU    → local features
    → FourierFeatures(2→base_dim, learnable freqs)      → positional features
    → Concat(local, pos) → Conv2d 1×1 projection        → pixel embeddings [B, D/2, H', W']
    → Downsample SLIC map to H' × W'
    → scatter_add / count  per superpixel                → avg tokens [B, K, D/2]
    → scatter_max          per superpixel                → max tokens [B, K, D/2]
    → Concat(avg, max)                                   → D-dim tokens [B, K, D]
    → Compute centroids from original SLIC map           → [B, K, 2] (cy, cx)

Multi-scale wrapper concatenates tokens from all levels with padding.

Reference: SuIT (Superpixel-tokenized Vision Transformer), ECCV 2024.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from torch_scatter import scatter_max

from util.misc import NestedTensor


# Fourier Features — Learnable Positional Encoding
class FourierFeatures(nn.Module):
    """Sinusoidal PE with learnable frequencies (Tancik et al., 2020).

    Produces positional features from 2D normalized coordinates.
    Unlike fixed sinusoidal PE, the frequency matrix B is a trainable
    parameter, allowing the model to learn vision-specific spatial patterns.

    Adapted from SuIT: suit.py/FourierFeatures.

    Args:
        pos_dim: input coordinate dimension (2 for 2D)
        ch: output channel dimension (must be even)
        sigma: std for random initialization of frequencies
        debug: print tensor stats when True
    """

    def __init__(
        self,
        pos_dim: int = 2,
        ch: int = 64,
        sigma: float = 10.0,
        debug: bool = False,
    ):
        super().__init__()
        assert ch % 2 == 0, f"ch must be even, got {ch}"
        self.debug = debug
        enc_dim = ch // 2
        # Learnable frequency matrix
        self.B = nn.Parameter(torch.randn(pos_dim, enc_dim) * sigma)

    def forward(self, pos: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pos: [..., pos_dim] normalized coordinates
        Returns:
            pe: [..., ch] positional encoding (sin/cos interleaved)
        """
        proj = torch.matmul(pos.float(), self.B)  # [..., enc_dim]
        pe = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)

        if self.debug:
            print(f"[FourierFeatures] Input: {pos.shape}, Output: {pe.shape}, "
                  f"mean={pe.mean():.4f}, std={pe.std():.4f}")

        return pe


# Learnable Tokenizer — Pixel Embedding + Superpixel Aggregation
class LearnableTokenizer(nn.Module):
    """Learnable pixel-level embedding + superpixel-aware aggregation.

    Replaces the handcrafted SuperpixelFeatureExtractorGPU with a fully
    differentiable pipeline. Gradients flow from loss through aggregation
    back to the conv weights, enabling task-specific feature learning.

    Adapted from SuIT: suit.py/SuperpixelVisionTransformer.

    Args:
        output_dim: transformer hidden dimension (D=256)
        base_dim: local feature dimension (D/4=64 by default)
        max_superpixels: padding limit per scale level
        downsample: conv stride for spatial downsampling (default: 2)
        debug: print tensor stats when True
    """

    def __init__(
        self,
        output_dim: int = 256,
        base_dim: int = 64,
        max_superpixels: int = 500,
        downsample: int = 2,
        debug: bool = False,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.base_dim = base_dim
        self.max_superpixels = max_superpixels
        self.downsample = downsample
        self.debug = debug

        # Aggregation dim: D/2 per method (avg, max), concat → D
        self.agg_dim = output_dim // 2

        # ── Stage 1: Pixel-level feature extraction ──
        # Local appearance features via learnable CNN
        self.local_features = nn.Sequential(
            nn.Conv2d(3, base_dim, kernel_size=7, stride=downsample,
                      padding=3, padding_mode='replicate'),
            nn.BatchNorm2d(base_dim),
            nn.GELU(),
        )

        # Learnable Fourier positional encoding
        self.pe = FourierFeatures(pos_dim=2, ch=base_dim, sigma=10.0)

        # Concat(local + pe) → 1×1 projection to agg_dim (= D/2)
        self.projection = nn.Conv2d(base_dim * 2, self.agg_dim, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        """Xavier init for projection, default init for conv/BN."""
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def _make_coords(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        """Create normalized coordinate grid matching feature map resolution.

        Follows SuIT convention: cartesian_prod(x, y) → [1, W, H, 2].

        Returns:
            coords: [1, W, H, 2] normalized coordinates in [0, 1]
        """
        x_coord = torch.arange(W, device=device).float() / max(W - 1, 1)
        y_coord = torch.arange(H, device=device).float() / max(H - 1, 1)
        coords = torch.cartesian_prod(x_coord, y_coord)  # [W*H, 2]
        return coords.reshape(1, W, H, 2)

    def _aggregate_superpixels(
        self, pixel_embed: torch.Tensor, slic_maps: torch.Tensor,
        B: int, K: int, device: torch.device,
    ) -> tuple:
        """Aggregate pixel embeddings into superpixel tokens via avg+max pooling.

        Uses a dummy bin (index K) for invalid pixels so they never
        contaminate valid superpixels.

        Args:
            pixel_embed: [B, agg_dim, fH, fW] pixel-level features
            slic_maps: [B, fH, fW] integer SLIC labels (downsampled)
            B: batch size
            K: max superpixels (padding limit)
            device: torch device

        Returns:
            tokens: [B, K, output_dim] superpixel tokens
            mask: [B, K] True = valid superpixel
        """
        agg_dim = self.agg_dim
        fH, fW = pixel_embed.shape[2], pixel_embed.shape[3]
        n_pixels = fH * fW

        # Flatten spatial dims
        pixel_flat = pixel_embed.view(B, agg_dim, n_pixels)  # [B, C, N]
        pixel_flat = pixel_flat.permute(0, 2, 1)             # [B, N, C]

        slic_flat = slic_maps.view(B, n_pixels)              # [B, N]
        valid_px = (slic_flat >= 0) & (slic_flat < K)

        # Remap invalid pixels to dummy bin K (isolated, discarded later)
        sp_idx = slic_flat.clone()
        sp_idx[~valid_px] = K

        valid_float = valid_px.unsqueeze(-1).float()  # [B, N, 1]
        sp_idx_c = sp_idx.unsqueeze(-1).expand(-1, -1, agg_dim)  # [B, N, C]
        sp_idx_1 = sp_idx.unsqueeze(-1)              # [B, N, 1]

        # ── Average pooling: scatter_add + manual count ──
        feat_sum = torch.zeros(B, K + 1, agg_dim, device=device)
        feat_sum.scatter_add_(1, sp_idx_c, pixel_flat * valid_float)
        feat_sum = feat_sum[:, :K, :]  # discard dummy bin

        counts = torch.zeros(B, K + 1, 1, device=device)
        counts.scatter_add_(1, sp_idx_1, valid_float)
        counts = counts[:, :K, :]     # discard dummy bin
        safe_counts = counts.clamp(min=1)

        avg_out = feat_sum / safe_counts  # [B, K, D/2]

        # ── Max pooling: scatter_max via torch_scatter ──
        # Reshape to [B*C, N] for scatter_max (requires 2D input)
        labels_bc = sp_idx.unsqueeze(1).expand(-1, agg_dim, -1)
        labels_bc = labels_bc.reshape(B * agg_dim, n_pixels).long()

        x_bc = pixel_flat.permute(0, 2, 1).reshape(B * agg_dim, n_pixels)
        # Mask invalid pixels to -inf so they can't win max
        invalid_bc = ~valid_px.unsqueeze(1).expand(-1, agg_dim, -1)
        invalid_bc = invalid_bc.reshape(B * agg_dim, n_pixels)
        x_bc = x_bc.clone()
        x_bc[invalid_bc] = float('-inf')

        max_flat, _ = scatter_max(x_bc, labels_bc, dim=1, dim_size=K + 1)
        max_flat = max_flat[:, :K]  # discard dummy bin, [B*C, K]
        max_out = max_flat.view(B, agg_dim, K).permute(0, 2, 1)  # [B, K, D/2]

        # Replace -inf (empty superpixels) with 0
        max_out = torch.where(
            max_out.isinf(), torch.zeros_like(max_out), max_out
        )

        # ── Concat avg + max → D-dim token ──
        tokens = torch.cat([avg_out, max_out], dim=2)  # [B, K, D]

        # Validity mask
        mask = (counts.squeeze(2) > 0)  # [B, K] True = valid
        tokens = tokens * mask.unsqueeze(-1).float()

        return tokens, mask

    def _compute_centroids(
        self, slic_maps: torch.Tensor, B: int, K: int,
        H: int, W: int, device: torch.device,
    ) -> torch.Tensor:
        """Compute normalized (cy, cx) centroids from original-resolution SLIC map.

        Returns:
            centroids: [B, K, 2] where dim[:,:,0]=cy, dim[:,:,1]=cx in [0, 1]
        """
        sp_flat = slic_maps.flatten(1)  # [B, H*W]
        valid = (sp_flat >= 0) & (sp_flat < K)
        sp_safe = sp_flat.clone()
        sp_safe[~valid] = 0

        valid_float = valid.unsqueeze(-1).float()
        sp_idx_2 = sp_safe.unsqueeze(-1).expand(-1, -1, 2)

        y_coords = torch.linspace(0, 1, H, device=device).view(1, H, 1).expand(B, H, W).flatten(1)
        x_coords = torch.linspace(0, 1, W, device=device).view(1, 1, W).expand(B, H, W).flatten(1)
        coords = torch.stack([y_coords, x_coords], dim=2) * valid_float  # [B, H*W, 2]

        ones = valid_float
        sp_idx_1 = sp_safe.unsqueeze(-1)
        counts = torch.zeros(B, K, 1, device=device)
        counts.scatter_add_(1, sp_idx_1, ones)
        safe_counts = counts.clamp(min=1)

        coord_sum = torch.zeros(B, K, 2, device=device)
        coord_sum.scatter_add_(1, sp_idx_2, coords)
        centroids = coord_sum / safe_counts

        return centroids

    def forward(
        self, images: torch.Tensor, slic_maps: torch.Tensor,
    ) -> tuple:
        """Extract learnable superpixel features from images.

        Args:
            images: [B, 3, H, W] normalized RGB images (on GPU)
            slic_maps: [B, H, W] integer superpixel IDs (pre-computed)

        Returns:
            features: [B, K, output_dim] superpixel tokens
            mask: [B, K] True for VALID superpixels
            centroids: [B, K, 2] normalized (cy, cx) centroids
        """
        B, _, H, W = images.shape
        K = self.max_superpixels
        device = images.device

        if self.debug:
            print(f"[LearnableTokenizer] Input: images={images.shape}, "
                  f"slic_maps={slic_maps.shape}")

        # ── Stage 1: Pixel-level embedding ──
        local_feat = self.local_features(images)  # [B, base_dim, H', W']
        _, _, fH, fW = local_feat.shape

        # Positional encoding from coordinate grid
        coords = self._make_coords(fH, fW, device)        # [1, W, H, 2]
        pe = self.pe(coords)                               # [1, W, H, base_dim]
        pe = pe.permute(0, 3, 2, 1)                        # [1, base_dim, H, W]
        pe = pe.expand(B, -1, -1, -1)                      # [B, base_dim, H, W]

        # Concat + project (SuIT best ablation: concat+proj > addition)
        pixel_embed = torch.cat([local_feat, pe], dim=1)   # [B, base_dim*2, H', W']
        pixel_embed = self.projection(pixel_embed)         # [B, D/2, H', W']

        # Downsample SLIC map to match feature map resolution
        slic_down = F.interpolate(
            slic_maps.unsqueeze(1).float(), size=(fH, fW), mode='nearest'
        ).squeeze(1).long()

        # ── Stage 2: Superpixel-aware aggregation (avg + max) ──
        tokens, mask = self._aggregate_superpixels(
            pixel_embed, slic_down, B, K, device,
        )

        # Centroids from original-resolution SLIC map (needed for box proposals)
        centroids = self._compute_centroids(slic_maps, B, K, H, W, device)

        if self.debug:
            print(f"[LearnableTokenizer] local_feat: {local_feat.shape}, "
                  f"mean={local_feat.mean():.4f}, std={local_feat.std():.4f}")
            print(f"[LearnableTokenizer] pixel_embed: {pixel_embed.shape}, "
                  f"mean={pixel_embed.mean():.4f}, std={pixel_embed.std():.4f}")
            print(f"[LearnableTokenizer] tokens: {tokens.shape}, "
                  f"mean={tokens.mean():.4f}, std={tokens.std():.4f}")
            print(f"[LearnableTokenizer] centroids: {centroids.shape}, "
                  f"mask valid: {mask.sum().item()}/{K*B}")

        return tokens, mask, centroids


# SLIC Feature Extractor (Multi-Scale Backbone wrapper for DINO)
class SLICFeatureExtractor(nn.Module):
    """SLIC Superpixel backbone for DINO (multi-scale, learnable).

    Wraps LearnableTokenizer with multi-scale support.
    Runs the tokenizer once per scale level, then concatenates
    tokens with padding to produce the output format expected by SLICTransformer.

    Output dict (unchanged interface from V1):
        tokens:       [bs, N_total, output_dim]  — learnable superpixel features
        centroids:    [bs, N_total, 2]            — normalized (cy, cx) in [0, 1]
        padding_mask: [bs, N_total]               — True = padded position
        level_counts: [num_levels]                — max superpixels per level

    Args:
        n_segments_per_level: list of SLIC n_segments for each scale
        compactness: SLIC compactness (for CPU fallback only)
        sigma: SLIC sigma (for CPU fallback only)
        output_dim: output feature dimension (must match DINO hidden_dim=256)
        max_superpixels_per_level: max superpixels per level for padding
        debug: if True, print tensor shapes and stats
    """

    def __init__(
        self,
        n_segments_per_level: List[int] = [400, 200, 100],
        compactness: float = 10.0,
        sigma: float = 1.0,
        output_dim: int = 256,
        max_superpixels_per_level: int = 500,
        debug: bool = False,
    ):
        super().__init__()
        self.n_segments_per_level = n_segments_per_level
        self.compactness = compactness
        self.sigma = sigma
        self.output_dim = output_dim
        self.debug = debug
        self.num_levels = len(n_segments_per_level)

        # Number of output channels per level (all same)
        self.num_channels = [output_dim] * self.num_levels

        # Learnable tokenizer (shared across levels)
        self.feature_extractor = LearnableTokenizer(
            output_dim=output_dim,
            base_dim=output_dim // 4,
            max_superpixels=max_superpixels_per_level,
            debug=debug,
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
            # Denormalize
            img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img_np = np.clip(img_np, 0, 1).astype(np.float32)
            segments = skimage_slic(
                img_np,
                n_segments=n_segments,
                compactness=self.compactness,
                sigma=self.sigma,
                start_label=0,
                channel_axis=2,
            )
            slic_maps[i] = torch.from_numpy(segments.astype(np.int64)).to(device)
        return slic_maps

    def forward(self, tensor_list: NestedTensor, targets=None) -> dict:
        """Forward pass: extract multi-scale learnable features as token sequences.

        Args:
            tensor_list: NestedTensor with:
                - tensors: [bs, 3, H, W] images
                - mask: [bs, H, W] padding mask
            targets: list of dicts, each optionally containing:
                - 'slic_maps': {n_seg: [H, W] tensor} — pre-computed per scale
        Returns:
            dict with:
                - tokens:       [bs, N_total, output_dim]
                - centroids:    [bs, N_total, 2]  — normalized (cy, cx)
                - padding_mask: [bs, N_total]      — True = padded
                - level_counts: list[int]          — max superpixels per level
        """
        images = tensor_list.tensors  # [bs, 3, H, W]
        B, _, H, W = images.shape
        device = images.device

        if self.debug:
            print(f"[SLICFeatureExtractor] Input: images={images.shape}")

        # Process each scale level
        level_features = []   # list of [B, K_lvl, output_dim]
        level_centroids = []  # list of [B, K_lvl, 2]
        level_masks = []      # list of [B, K_lvl] (True = valid)
        level_counts = []     # max tokens per level

        for lvl, n_seg in enumerate(self.n_segments_per_level):
            # Get slic_maps for this level
            has_precomputed = (
                targets is not None
                and len(targets) > 0
                and 'slic_maps' in targets[0]
                and n_seg in targets[0]['slic_maps']
            )

            if has_precomputed:
                # Use pre-computed maps from dataset
                slic_maps = torch.stack([
                    self._resize_slic_map(t['slic_maps'][n_seg].to(device), H, W)
                    for t in targets
                ])  # [B, H, W]
            else:
                # Fallback: run SLIC on-the-fly (CPU)
                slic_maps = self._run_slic_cpu(images, n_seg)

            # Learnable feature extraction + aggregation
            features, mask, centroids = self.feature_extractor(images, slic_maps)
            # features: [B, max_superpixels, output_dim]
            # mask: [B, max_superpixels] (True = valid)
            # centroids: [B, max_superpixels, 2]

            # Find the actual max valid count across the batch for this level
            valid_counts = mask.sum(dim=1)  # [B]
            max_valid = int(valid_counts.max().item())
            max_valid = max(max_valid, 1)  # at least 1 token

            # Trim to actual size needed
            features = features[:, :max_valid]
            centroids = centroids[:, :max_valid]
            mask = mask[:, :max_valid]

            level_features.append(features)
            level_centroids.append(centroids)
            level_masks.append(mask)
            level_counts.append(max_valid)

        # Concatenate all levels
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
            print(f"[SLICFeatureExtractor] Output: tokens={all_tokens.shape}, "
                  f"centroids={all_centroids.shape}, mask={all_masks.shape}, "
                  f"level_counts={level_counts}")
            print(f"[SLICFeatureExtractor] tokens mean={all_tokens.mean():.4f}, "
                  f"std={all_tokens.std():.4f}")

        return {
            'tokens': all_tokens,        # [bs, N_total, output_dim]
            'centroids': all_centroids,   # [bs, N_total, 2] — (cy, cx) normalized
            'padding_mask': all_masks,    # [bs, N_total] — True = padded
            'level_counts': level_counts, # list[int] — per-level token counts
        }
