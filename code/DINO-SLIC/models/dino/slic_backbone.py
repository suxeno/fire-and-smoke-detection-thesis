# ------------------------------------------------------------------------
# DINO-SLIC
# GPU-based Superpixel Feature Extraction Backbone (Multi-Scale)
# Replaces CNN backbone with handcrafted superpixel features
# ALL operations run on GPU via scatter_add — zero CPU transfers
# Ported from DETR-SLIC's SuperpixelFeatureExtractorGPU
# ------------------------------------------------------------------------

"""
SLIC Superpixel Backbone Module (GPU-only, Multi-Scale).

Pipeline:
    Pre-computed SLIC maps (per scale) + Image (on GPU)
    → GPU Feature Extraction (handcrafted features per superpixel)
    → Geometry path: shape+spatial+structural (18D) → normalize [0,1] → MLP → 64D
    → Appearance path: color+texture+appearance-relational (118D) → raw
    → Combined (182D) → LayerNorm → Linear(182→256) → LayerNorm
    → Multi-scale Superpixel Token Sequences + Normalized Centroids

Feature groups:
    Appearance (118D):
        Color (81):    6 stats × 3ch × 3 spaces + 9 entropy + 24 hist + 3 dominant
        Texture (35):  4 Sobel + 4 GLCM + 16 LBP + 9 HOG + 2 grad mag
        Relational-appearance (2): color_diff + texture_diff
    Geometry (18D → 64D via GeometryMLP):
        Shape (12):    area + perimeter + compactness + var_y + var_x + cov
                       + eccentricity + 5 Hu
        Spatial (4):   cx + cy + dist_center + angle_center
        Relational-structural (2): boundary_strength + degree
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
from util.misc import NestedTensor


# =============================================================================
# Constants
# =============================================================================
GEOMETRY_DIM = 18     # shape(12) + spatial(4) + boundary_strength(1) + degree(1)
APPEARANCE_DIM = 118  # color(81) + texture(35) + color_diff(1) + texture_diff(1)
GEO_EMBED_DIM = 64    # geometry MLP output dimension
COMBINED_DIM = APPEARANCE_DIM + GEO_EMBED_DIM  # 182 (input to final projection)


# =============================================================================
# Color Space Conversions (GPU, differentiable)
# =============================================================================
def rgb_to_lab(rgb: torch.Tensor) -> torch.Tensor:
    """Convert RGB [B,3,H,W] in [0,1] to Lab. L:[0,1], a,b:[-1,1]."""
    mask = rgb > 0.04045
    rgb_lin = torch.where(mask, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
    r, g, b = rgb_lin[:, 0:1], rgb_lin[:, 1:2], rgb_lin[:, 2:3]

    x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
    y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041

    x = x / 0.95047
    z = z / 1.08883

    eps = 0.008856
    kappa = 903.3
    fx = torch.where(x > eps, x ** (1 / 3), (kappa * x + 16) / 116)
    fy = torch.where(y > eps, y ** (1 / 3), (kappa * y + 16) / 116)
    fz = torch.where(z > eps, z ** (1 / 3), (kappa * z + 16) / 116)

    L = (116 * fy - 16) / 100.0
    a = 500 * (fx - fy) / 128.0
    b_lab = 200 * (fy - fz) / 128.0

    return torch.cat([L, a, b_lab], dim=1)


def rgb_to_hsv(rgb: torch.Tensor) -> torch.Tensor:
    """Convert RGB [B,3,H,W] in [0,1] to HSV in [0,1]."""
    r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
    max_rgb, _ = rgb.max(dim=1, keepdim=True)
    min_rgb, _ = rgb.min(dim=1, keepdim=True)
    diff = max_rgb - min_rgb + 1e-8

    v = max_rgb
    s = torch.where(max_rgb > 1e-8, diff / (max_rgb + 1e-8), torch.zeros_like(max_rgb))

    h = torch.zeros_like(max_rgb)
    h = torch.where(max_rgb == r, ((g - b) / diff) % 6, h)
    h = torch.where(max_rgb == g, (b - r) / diff + 2, h)
    h = torch.where(max_rgb == b, (r - g) / diff + 4, h)
    h = (h / 6.0).clamp(0, 1)

    return torch.cat([h, s, v], dim=1)


# =============================================================================
# Geometry MLP — Dedicated embedding for shape/spatial/structural features
# =============================================================================
class GeometryMLP(nn.Module):
    """2-layer MLP that normalizes and embeds geometry features.

    Geometry features (shape 12D + spatial 4D + structural-relational 2D = 18D)
    are first normalized to [0, 1] via feature-specific rescaling, then embedded
    via a learned MLP to produce a compact geometry representation.

    Architecture:
        normalize [0,1] → Linear(18→64) → GELU → LayerNorm(64)
                        → Linear(64→64) → LayerNorm(64)

    Args:
        input_dim: raw geometry feature dimension (default: 18)
        embed_dim: output embedding dimension (default: 64)
        debug: print tensor stats when True
    """

    def __init__(
        self,
        input_dim: int = GEOMETRY_DIM,
        embed_dim: int = GEO_EMBED_DIM,
        debug: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.debug = debug

        # 2-layer MLP: input_dim → embed_dim → embed_dim
        self.linear1 = nn.Linear(input_dim, embed_dim)
        self.act = nn.GELU()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.linear2 = nn.Linear(embed_dim, embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def _normalize_geometry(self, geo: torch.Tensor) -> torch.Tensor:
        """Feature-specific normalization to [0, 1].

        Input ordering (18D):
            [0]  area           [0, ~0.01]  → /0.02
            [1]  perimeter      [0, ~0.5]   → /0.6
            [2]  compactness    [0, 1]      → passthrough
            [3]  var_y          [0, ~0.08]  → /0.1
            [4]  var_x          [0, ~0.08]  → /0.1
            [5]  cov_yx         [-0.05,0.05]→ /0.1 + 0.5
            [6]  eccentricity   [0, 1]      → passthrough
            [7:12] hu_log (5D)  [-20, 20]   → /40 + 0.5
            [12] cx             [0, 1]      → passthrough
            [13] cy             [0, 1]      → passthrough
            [14] dist_center    [0, ~0.7]   → /0.8
            [15] angle_center   [-π, π]     → /(2π) + 0.5
            [16] boundary_str   [0, ~1]     → passthrough
            [17] degree         [0, ~1]     → passthrough

        Returns:
            Normalized tensor, same shape as input, values in [0, 1].
        """
        out = geo.clone()
        out[:, :, 0]    = (geo[:, :, 0] / 0.02)                         # area
        out[:, :, 1]    = (geo[:, :, 1] / 0.6)                          # perimeter
        # [2] compactness — already [0,1]
        out[:, :, 3]    = (geo[:, :, 3] / 0.1)                          # var_y
        out[:, :, 4]    = (geo[:, :, 4] / 0.1)                          # var_x
        out[:, :, 5]    = (geo[:, :, 5] / 0.1 + 0.5)                    # cov_yx
        # [6] eccentricity — already [0,1]
        out[:, :, 7:12] = (geo[:, :, 7:12] / 40.0 + 0.5)               # hu_log
        # [12] cx, [13] cy — already [0,1]
        out[:, :, 14]   = (geo[:, :, 14] / 0.8)                         # dist_center
        out[:, :, 15]   = (geo[:, :, 15] / (2.0 * math.pi) + 0.5)      # angle_center
        # [16] boundary_strength, [17] degree — already ~[0,1]
        return out.clamp(0.0, 1.0)

    def forward(self, geo_features: torch.Tensor) -> torch.Tensor:
        """Normalize and embed geometry features.

        Args:
            geo_features: [B, K, 18] raw geometry features
        Returns:
            geo_embed: [B, K, embed_dim] learned geometry embedding
        """
        if self.debug:
            print(f"[GeometryMLP] Input: {geo_features.shape}, "
                  f"mean={geo_features.mean():.4f}, std={geo_features.std():.4f}")

        x = self._normalize_geometry(geo_features)

        if self.debug:
            print(f"[GeometryMLP] After norm: mean={x.mean():.4f}, "
                  f"std={x.std():.4f}, min={x.min():.4f}, max={x.max():.4f}")

        x = self.norm1(self.act(self.linear1(x)))
        x = self.norm2(self.linear2(x))

        if self.debug:
            print(f"[GeometryMLP] Output: {x.shape}, "
                  f"mean={x.mean():.4f}, std={x.std():.4f}")

        return x


# =============================================================================
# GPU Feature Extractor
# =============================================================================
class SuperpixelFeatureExtractorGPU(nn.Module):
    """Extracts handcrafted features from superpixels entirely on GPU.

    Feature extraction produces appearance (118D) and geometry (18D) features.
    Geometry features are embedded via a dedicated GeometryMLP (18D → 64D)
    before being concatenated with appearance features. The combined 182D
    vector is then projected to the transformer hidden dimension.

    Uses scatter_add for all per-superpixel aggregation — no Python loops,
    no CPU transfers during the forward pass.

    Args:
        hidden_dim: transformer hidden dimension (projection output)
        max_superpixels: padding limit for the token sequence
        geo_embed_dim: geometry MLP output dimension
        debug: print tensor shapes and stats when True
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        max_superpixels: int = 300,
        geo_embed_dim: int = GEO_EMBED_DIM,
        debug: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_superpixels = max_superpixels
        self.debug = debug

        # Geometry embedding: 18D → geo_embed_dim (64D)
        self.geometry_mlp = GeometryMLP(
            input_dim=GEOMETRY_DIM,
            embed_dim=geo_embed_dim,
            debug=debug,
        )

        # Combined projection: appearance(118D) + geo_embed(64D) = 182D → hidden_dim
        combined_dim = APPEARANCE_DIM + geo_embed_dim
        self.input_norm = nn.LayerNorm(combined_dim)
        self.projection = nn.Linear(combined_dim, hidden_dim)
        self.output_norm = nn.LayerNorm(hidden_dim)

        # Sobel kernels (registered as buffers for GPU persistence)
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                               dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                               dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

        # LBP 8-neighbor offsets
        lbp_offsets = torch.tensor([
            [-1, -1], [-1, 0], [-1, 1],
            [0, 1], [1, 1], [1, 0],
            [1, -1], [0, -1]
        ], dtype=torch.long)
        self.register_buffer('lbp_offsets', lbp_offsets)

        self._init_weights()

    def _init_weights(self):
        # Final projection
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)
        nn.init.constant_(self.input_norm.weight, 1.0)
        nn.init.constant_(self.input_norm.bias, 0.0)
        nn.init.constant_(self.output_norm.weight, 1.0)
        nn.init.constant_(self.output_norm.bias, 0.0)
        # GeometryMLP
        nn.init.xavier_uniform_(self.geometry_mlp.linear1.weight)
        nn.init.zeros_(self.geometry_mlp.linear1.bias)
        nn.init.xavier_uniform_(self.geometry_mlp.linear2.weight)
        nn.init.zeros_(self.geometry_mlp.linear2.bias)
        nn.init.constant_(self.geometry_mlp.norm1.weight, 1.0)
        nn.init.constant_(self.geometry_mlp.norm1.bias, 0.0)
        nn.init.constant_(self.geometry_mlp.norm2.weight, 1.0)
        nn.init.constant_(self.geometry_mlp.norm2.bias, 0.0)

    # -----------------------------------------------------------------
    # Color features (81D)
    # -----------------------------------------------------------------
    def _compute_color_features(
        self, images, sp_flat, sp_idx_3, sp_idx_1, valid_float, safe_counts,
        B, K, device
    ):
        """Compute 81D color features on GPU.

        Per color space (RGB, Lab, HSV) × (mean, std, range, skew, kurt) × 3ch = 45D
        + entropy per channel per space = 9D
        + RGB histogram 8×3 = 24D
        + dominant color = 3D
        Total: 81D
        """
        features_list = []

        lab = rgb_to_lab(images)
        hsv = rgb_to_hsv(images)

        for color_img in [images, lab, hsv]:
            # Flatten: [B, 3, H, W] -> [B, H*W, 3]
            c_flat = color_img.flatten(2).transpose(1, 2) * valid_float

            # Raw moments via scatter_add
            c_sum = torch.zeros(B, K, 3, device=device)
            c_sum.scatter_add_(1, sp_idx_3, c_flat)
            c_mean = c_sum / safe_counts  # mean

            c_sq = torch.zeros(B, K, 3, device=device)
            c_sq.scatter_add_(1, sp_idx_3, c_flat ** 2)
            c_var = (c_sq / safe_counts - c_mean ** 2).clamp(min=0)
            c_std = torch.sqrt(c_var + 1e-6)  # std
            c_range = 2.0 * c_std  # range ≈ 2σ

            # 3rd moment for skewness
            c_cu = torch.zeros(B, K, 3, device=device)
            c_cu.scatter_add_(1, sp_idx_3, c_flat ** 3)
            mu3 = c_cu / safe_counts - 3 * c_mean * c_sq / safe_counts + 2 * c_mean ** 3
            c_skew = (mu3 / (c_std ** 3 + 1e-8)).clamp(-10, 10)

            # 4th moment for kurtosis
            c_qu = torch.zeros(B, K, 3, device=device)
            c_qu.scatter_add_(1, sp_idx_3, c_flat ** 4)
            mu4 = (c_qu / safe_counts
                   - 4 * c_mean * c_cu / safe_counts
                   + 6 * c_mean ** 2 * c_sq / safe_counts
                   - 3 * c_mean ** 4)
            c_kurt = (mu4 / (c_var ** 2 + 1e-8) - 3.0).clamp(-10, 10)

            features_list.extend([c_mean, c_std, c_range, c_skew, c_kurt])  # 5×3 = 15D

        # --- Entropy per channel per color space (9D) ---
        HIST_BINS = 8
        for color_idx, color_img in enumerate([images, lab, hsv]):
            c_flat = color_img.flatten(2).transpose(1, 2) * valid_float
            entropy_ch = []
            for ch in range(3):
                ch_vals = c_flat[:, :, ch:ch+1]  # [B, H*W, 1]
                # Build per-superpixel histogram
                hist = torch.zeros(B, K, HIST_BINS, device=device)
                for bin_i in range(HIST_BINS):
                    lo = bin_i / HIST_BINS
                    hi = (bin_i + 1) / HIST_BINS
                    if bin_i == HIST_BINS - 1:
                        in_bin = ((ch_vals >= lo) & (ch_vals <= hi)).float() * valid_float
                    else:
                        in_bin = ((ch_vals >= lo) & (ch_vals < hi)).float() * valid_float
                    hist[:, :, bin_i:bin_i+1].scatter_add_(1, sp_idx_1, in_bin)
                # Normalize to probabilities
                hist_sum = hist.sum(dim=2, keepdim=True).clamp(min=1)
                p = hist / hist_sum
                ent = -(p * torch.log2(p + 1e-10)).sum(dim=2, keepdim=True)
                entropy_ch.append(ent)
            features_list.append(torch.cat(entropy_ch, dim=2))  # 3D

        # --- RGB histogram (8 bins × 3 channels = 24D) ---
        rgb_flat = images.flatten(2).transpose(1, 2) * valid_float
        for ch in range(3):
            ch_vals = rgb_flat[:, :, ch:ch+1]
            hist = torch.zeros(B, K, HIST_BINS, device=device)
            for bin_i in range(HIST_BINS):
                lo = bin_i / HIST_BINS
                hi = (bin_i + 1) / HIST_BINS
                if bin_i == HIST_BINS - 1:
                    in_bin = ((ch_vals >= lo) & (ch_vals <= hi)).float() * valid_float
                else:
                    in_bin = ((ch_vals >= lo) & (ch_vals < hi)).float() * valid_float
                hist[:, :, bin_i:bin_i+1].scatter_add_(1, sp_idx_1, in_bin)
            hist_sum = hist.sum(dim=2, keepdim=True).clamp(min=1)
            hist = hist / hist_sum
            features_list.append(hist)  # 8D per channel

        # --- Dominant color (3D) ---
        rgb_sum = torch.zeros(B, K, 3, device=device)
        rgb_sum.scatter_add_(1, sp_idx_3, rgb_flat)
        dominant_color = rgb_sum / safe_counts
        features_list.append(dominant_color)

        return torch.cat(features_list, dim=2)  # 81D

    # -----------------------------------------------------------------
    # Texture features (35D)
    # -----------------------------------------------------------------
    def _compute_texture_features(
        self, images, sp_flat, sp_idx_1, valid_float, safe_counts,
        B, K, H, W, device
    ):
        """Compute 35D texture features on GPU.

        Sobel gradient stats: 4D
        GLCM approximation: 4D
        LBP histogram (16-bin): 16D
        HOG histogram (9-bin): 9D
        Gradient magnitude mean+std: 2D
        Total: 35D
        """
        gray = images.mean(dim=1, keepdim=True)  # [B, 1, H, W]
        grad_x = F.conv2d(gray, self.sobel_x, padding=1)
        grad_y = F.conv2d(gray, self.sobel_y, padding=1)

        # --- Sobel gradient stats (4D): mean_x, std_x, mean_y, std_y ---
        gx_flat = grad_x.flatten(2).transpose(1, 2) * valid_float
        gy_flat = grad_y.flatten(2).transpose(1, 2) * valid_float

        gx_sum = torch.zeros(B, K, 1, device=device)
        gx_sum.scatter_add_(1, sp_idx_1, gx_flat)
        gx_mean = gx_sum / safe_counts

        gx_sq = torch.zeros(B, K, 1, device=device)
        gx_sq.scatter_add_(1, sp_idx_1, gx_flat ** 2)
        gx_std = torch.sqrt((gx_sq / safe_counts - gx_mean ** 2).clamp(min=0) + 1e-6)

        gy_sum = torch.zeros(B, K, 1, device=device)
        gy_sum.scatter_add_(1, sp_idx_1, gy_flat)
        gy_mean = gy_sum / safe_counts

        gy_sq = torch.zeros(B, K, 1, device=device)
        gy_sq.scatter_add_(1, sp_idx_1, gy_flat ** 2)
        gy_std = torch.sqrt((gy_sq / safe_counts - gy_mean ** 2).clamp(min=0) + 1e-6)

        sobel_features = torch.cat([gx_mean, gx_std, gy_mean, gy_std], dim=2)  # 4D

        # --- GLCM approximation (4D): contrast, energy, homogeneity, entropy ---
        gray_q = (gray.flatten(2).transpose(1, 2) * 15).clamp(0, 15) * valid_float
        gq_sum = torch.zeros(B, K, 1, device=device)
        gq_sum.scatter_add_(1, sp_idx_1, gray_q)
        gq_mean = gq_sum / safe_counts

        gq_sq = torch.zeros(B, K, 1, device=device)
        gq_sq.scatter_add_(1, sp_idx_1, gray_q ** 2)
        contrast = (gq_sq / safe_counts - gq_mean ** 2).clamp(min=0) / 225.0
        energy = 1.0 / (1.0 + contrast * 225.0)
        g_std = torch.sqrt(contrast * 225.0 + 1e-6)
        homogeneity = 1.0 / (1.0 + g_std / 15.0)
        entropy = g_std / 15.0

        glcm_features = torch.cat([contrast, energy, homogeneity, entropy], dim=2)  # 4D

        # --- LBP histogram (16-bin) ---
        gray_pad = F.pad(gray, (1, 1, 1, 1), mode='replicate')
        lbp_code = torch.zeros(B, 1, H, W, device=device)
        for i, (dy, dx) in enumerate(self.lbp_offsets):
            neighbor = gray_pad[:, :, 1+dy:H+1+dy, 1+dx:W+1+dx]
            lbp_code = lbp_code + (neighbor >= gray).float() * (2 ** i)
        lbp_bin = (lbp_code / 16.0).floor().clamp(0, 15)
        lbp_flat = lbp_bin.flatten(2).transpose(1, 2)  # [B, H*W, 1]

        lbp_hist = torch.zeros(B, K, 16, device=device)
        for bin_i in range(16):
            in_bin = ((lbp_flat >= bin_i) & (lbp_flat < bin_i + 1)).float() * valid_float
            lbp_hist[:, :, bin_i:bin_i+1].scatter_add_(1, sp_idx_1, in_bin)
        lbp_hist = lbp_hist / safe_counts.clamp(min=1)  # 16D

        # --- HOG histogram (9-bin) + gradient magnitude mean/std (2D) ---
        magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)
        orientation = torch.atan2(grad_y, grad_x)  # [-π, π]
        orientation = (orientation + math.pi) / (2 * math.pi) * 9  # [0, 9)
        orientation = orientation.floor().clamp(0, 8)  # bins 0-8

        orient_flat = orientation.flatten(2).transpose(1, 2)
        mag_flat = magnitude.flatten(2).transpose(1, 2) * valid_float

        HOG_BINS = 9
        hog_hist = torch.zeros(B, K, HOG_BINS, device=device)
        for bin_i in range(HOG_BINS):
            in_bin = ((orient_flat >= bin_i) & (orient_flat < bin_i + 1)).float()
            weighted = in_bin * mag_flat
            hog_hist[:, :, bin_i:bin_i+1].scatter_add_(1, sp_idx_1, weighted)
        hog_sum = hog_hist.sum(dim=2, keepdim=True).clamp(min=1e-6)
        hog_hist = hog_hist / hog_sum  # 9D

        # Gradient magnitude mean + std (2D)
        mag_sum = torch.zeros(B, K, 1, device=device)
        mag_sum.scatter_add_(1, sp_idx_1, mag_flat)
        mag_mean = mag_sum / safe_counts

        mag_sq = torch.zeros(B, K, 1, device=device)
        mag_sq.scatter_add_(1, sp_idx_1, mag_flat ** 2)
        mag_std = torch.sqrt((mag_sq / safe_counts - mag_mean ** 2).clamp(min=0) + 1e-6)

        return torch.cat([
            sobel_features,  # 4D
            glcm_features,   # 4D
            lbp_hist,        # 16D
            hog_hist,        # 9D
            mag_mean,        # 1D
            mag_std,         # 1D
        ], dim=2)  # 35D

    # -----------------------------------------------------------------
    # Shape features (12D)
    # -----------------------------------------------------------------
    def _compute_shape_features(
        self, sp_flat, sp_idx_1, sp_idx_2, valid_float, safe_counts, counts,
        sp_map, B, K, H, W, device
    ):
        """Compute 12D shape features on GPU.

        area, perimeter, compactness, var_y, var_x, cov_yx, eccentricity, 5 Hu moments
        Total: 12D
        """
        # Coordinate grids (normalized to [0,1])
        y_coords = torch.linspace(0, 1, H, device=device).view(1, H, 1).expand(B, H, W).flatten(1)
        x_coords = torch.linspace(0, 1, W, device=device).view(1, 1, W).expand(B, H, W).flatten(1)
        coords = torch.stack([y_coords, x_coords], dim=2) * valid_float  # [B, H*W, 2]

        # Centroids
        coord_sum = torch.zeros(B, K, 2, device=device)
        coord_sum.scatter_add_(1, sp_idx_2, coords)
        centroids = coord_sum / safe_counts

        # Area (normalized)
        area = counts / (H * W)

        # Perimeter: count boundary pixels (where sp_id != any 4-neighbor)
        sp_map_pad = F.pad(sp_map.unsqueeze(1).float(), (1, 1, 1, 1), mode='replicate').squeeze(1).long()
        boundary = (
            (sp_map != sp_map_pad[:, 1:-1, 2:]) |   # right
            (sp_map != sp_map_pad[:, 2:, 1:-1]) |   # down
            (sp_map != sp_map_pad[:, 1:-1, :-2]) |  # left
            (sp_map != sp_map_pad[:, :-2, 1:-1])     # up
        )
        boundary_flat = (boundary.flatten(1).unsqueeze(-1).float() * valid_float)
        perimeter = torch.zeros(B, K, 1, device=device)
        perimeter.scatter_add_(1, sp_idx_1, boundary_flat)
        perimeter_norm = perimeter / (H + W)  # normalize

        # Compactness: 4πA / P²
        compactness = (4 * math.pi * counts) / (perimeter ** 2 + 1e-6)
        compactness = compactness.clamp(0, 1)

        # Second moments for covariance
        y_flat = y_coords.unsqueeze(-1) * valid_float
        x_flat = x_coords.unsqueeze(-1) * valid_float

        yy = torch.zeros(B, K, 1, device=device)
        yy.scatter_add_(1, sp_idx_1, (y_flat ** 2))
        xx = torch.zeros(B, K, 1, device=device)
        xx.scatter_add_(1, sp_idx_1, (x_flat ** 2))
        yx = torch.zeros(B, K, 1, device=device)
        yx.scatter_add_(1, sp_idx_1, y_flat * x_flat)

        cy = centroids[:, :, 0:1]  # [B, K, 1]
        cx = centroids[:, :, 1:2]

        var_y = (yy / safe_counts - cy ** 2).clamp(min=0)
        var_x = (xx / safe_counts - cx ** 2).clamp(min=0)
        cov_yx = yx / safe_counts - cy * cx

        # Eccentricity from covariance matrix eigenvalues
        disc = torch.sqrt(((var_y - var_x) ** 2 + 4 * cov_yx ** 2).clamp(min=0) + 1e-10)
        lambda_max = 0.5 * (var_y + var_x + disc)
        lambda_min = 0.5 * (var_y + var_x - disc).clamp(min=0)
        eccentricity = torch.sqrt(1.0 - lambda_min / (lambda_max + 1e-8)).clamp(0, 1)

        # Hu moments (first 5 of 7)
        y3 = torch.zeros(B, K, 1, device=device)
        y3.scatter_add_(1, sp_idx_1, y_flat ** 3)
        x3 = torch.zeros(B, K, 1, device=device)
        x3.scatter_add_(1, sp_idx_1, x_flat ** 3)
        y2x = torch.zeros(B, K, 1, device=device)
        y2x.scatter_add_(1, sp_idx_1, (y_flat ** 2) * x_flat)
        yx2 = torch.zeros(B, K, 1, device=device)
        yx2.scatter_add_(1, sp_idx_1, y_flat * (x_flat ** 2))

        # Central moments from raw
        mu_20 = var_y
        mu_02 = var_x
        mu_11 = cov_yx
        mu_30 = y3 / safe_counts - 3 * cy * yy / safe_counts + 2 * cy ** 3
        mu_03 = x3 / safe_counts - 3 * cx * xx / safe_counts + 2 * cx ** 3
        mu_21 = y2x / safe_counts - 2 * cy * yx / safe_counts - cx * yy / safe_counts + 2 * cy ** 2 * cx
        mu_12 = yx2 / safe_counts - 2 * cx * yx / safe_counts - cy * xx / safe_counts + 2 * cx ** 2 * cy

        # Normalized central moments
        eta_20 = mu_20
        eta_02 = mu_02
        eta_11 = mu_11
        eta_30 = mu_30 / (safe_counts.squeeze(-1).unsqueeze(-1) ** 0.5 + 1e-8)
        eta_03 = mu_03 / (safe_counts.squeeze(-1).unsqueeze(-1) ** 0.5 + 1e-8)
        eta_21 = mu_21 / (safe_counts.squeeze(-1).unsqueeze(-1) ** 0.5 + 1e-8)
        eta_12 = mu_12 / (safe_counts.squeeze(-1).unsqueeze(-1) ** 0.5 + 1e-8)

        # Hu moments (log-transformed)
        hu1 = eta_20 + eta_02
        hu2 = (eta_20 - eta_02) ** 2 + 4 * eta_11 ** 2
        hu3 = (eta_30 - 3 * eta_12) ** 2 + (3 * eta_21 - eta_03) ** 2
        hu4 = (eta_30 + eta_12) ** 2 + (eta_21 + eta_03) ** 2
        hu5 = ((eta_30 - 3 * eta_12) * (eta_30 + eta_12) *
               ((eta_30 + eta_12) ** 2 - 3 * (eta_21 + eta_03) ** 2) +
               (3 * eta_21 - eta_03) * (eta_21 + eta_03) *
               (3 * (eta_30 + eta_12) ** 2 - (eta_21 + eta_03) ** 2))

        hu_stack = torch.cat([hu1, hu2, hu3, hu4, hu5], dim=2)
        hu_log = -torch.sign(hu_stack) * torch.log10(hu_stack.abs() + 1e-10)
        hu_log = hu_log.clamp(-20, 20)

        return torch.cat([
            area,           # 1D
            perimeter_norm, # 1D
            compactness,    # 1D
            var_y,          # 1D
            var_x,          # 1D
            cov_yx,         # 1D
            eccentricity,   # 1D
            hu_log,         # 5D
        ], dim=2), centroids  # 12D total, plus centroids for later use

    # -----------------------------------------------------------------
    # Spatial features (4D)
    # -----------------------------------------------------------------
    def _compute_spatial_features(self, centroids, B, K, device):
        """Compute 4D spatial features from centroids."""
        cy = centroids[:, :, 0:1]
        cx = centroids[:, :, 1:2]
        dist = torch.sqrt((cy - 0.5) ** 2 + (cx - 0.5) ** 2)
        angle = torch.atan2(cy - 0.5, cx - 0.5)
        return torch.cat([cx, cy, dist, angle], dim=2)  # 4D

    # -----------------------------------------------------------------
    # Relational features (4D)
    # -----------------------------------------------------------------
    def _compute_relational_features(
        self, sp_map, rgb_mean, grad_mag_mean, centroids, mask,
        B, K, H, W, device
    ):
        """Compute relational features via boundary scatter.

        Returns two tensors, split by feature group:
            appear_rel: [B, K, 2] — color_diff, texture_diff (appearance)
            geo_rel:    [B, K, 2] — boundary_strength, degree (structural/geometry)
        """
        sp_map_pad = F.pad(sp_map.unsqueeze(1).float(), (1, 1, 1, 1), mode='replicate').squeeze(1).long()

        neighbors_r = sp_map_pad[:, 1:-1, 2:]
        neighbors_d = sp_map_pad[:, 2:, 1:-1]

        boundary_r = (sp_map != neighbors_r) & (sp_map >= 0) & (neighbors_r >= 0)
        boundary_d = (sp_map != neighbors_d) & (sp_map >= 0) & (neighbors_d >= 0)

        color_diff_acc = torch.zeros(B, K, 1, device=device)
        texture_diff_acc = torch.zeros(B, K, 1, device=device)
        neighbor_count = torch.zeros(B, K, 1, device=device)

        sp_flat = sp_map.flatten(1)
        neigh_r_flat = neighbors_r.flatten(1)
        neigh_d_flat = neighbors_d.flatten(1)
        boundary_mask = (boundary_r | boundary_d).flatten(1)

        for b in range(B):
            bnd_px = boundary_mask[b].nonzero(as_tuple=True)[0]
            if len(bnd_px) == 0:
                continue

            sp_ids = sp_flat[b, bnd_px]
            nr = neigh_r_flat[b, bnd_px]

            valid = (sp_ids != nr) & (sp_ids >= 0) & (sp_ids < K) & (nr >= 0) & (nr < K)
            if not valid.any():
                continue

            sv = sp_ids[valid]
            nv = nr[valid]

            c_diff = (rgb_mean[b, sv] - rgb_mean[b, nv]).abs().mean(dim=1, keepdim=True)
            t_diff = (grad_mag_mean[b, sv] - grad_mag_mean[b, nv]).abs()

            color_diff_acc[b].scatter_add_(0, sv.unsqueeze(1), c_diff)
            texture_diff_acc[b].scatter_add_(0, sv.unsqueeze(1), t_diff)
            neighbor_count[b].scatter_add_(0, sv.unsqueeze(1), torch.ones_like(c_diff))

        safe_nc = neighbor_count.clamp(min=1)
        mean_color_diff = color_diff_acc / safe_nc
        mean_texture_diff = texture_diff_acc / safe_nc
        boundary_strength = mean_texture_diff
        degree = neighbor_count / 4.0

        appear_rel = torch.cat([mean_color_diff, mean_texture_diff], dim=2)  # 2D
        geo_rel = torch.cat([boundary_strength, degree], dim=2)              # 2D
        return appear_rel, geo_rel

    # -----------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------
    def forward(self, images: torch.Tensor, slic_maps: torch.Tensor):
        """Extract 136-dim features per superpixel, all on GPU.

        Args:
            images: [B, 3, H, W] normalized RGB images (on GPU)
            slic_maps: [B, H, W] integer superpixel IDs (pre-computed)

        Returns:
            features: [B, K, hidden_dim] projected superpixel features
            mask: [B, K] True for VALID superpixels
            centroids: [B, K, 2] normalized (cy, cx) centroids
        """
        B, _, H, W = images.shape
        K = self.max_superpixels
        device = images.device

        # Denormalize images to [0,1] range for feature extraction
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        images_01 = (images * std + mean).clamp(0, 1)

        # Flatten superpixel map
        sp_flat = slic_maps.flatten(1)  # [B, H*W]
        valid_pixels = (sp_flat >= 0) & (sp_flat < K)
        sp_flat_safe = sp_flat.clone()
        sp_flat_safe[~valid_pixels] = 0

        # Index tensors for scatter_add
        sp_idx_1 = sp_flat_safe.unsqueeze(-1)                  # [B, H*W, 1]
        sp_idx_2 = sp_idx_1.expand(-1, -1, 2)                 # [B, H*W, 2]
        sp_idx_3 = sp_idx_1.expand(-1, -1, 3)                 # [B, H*W, 3]
        valid_float = valid_pixels.unsqueeze(-1).float()       # [B, H*W, 1]

        # Counts
        ones = valid_float
        counts = torch.zeros(B, K, 1, device=device)
        counts.scatter_add_(1, sp_idx_1, ones)
        safe_counts = counts.clamp(min=1)
        mask = (counts.squeeze(2) > 0)  # [B, K] True = valid

        # ── Compute all features ──
        color_feat = self._compute_color_features(
            images_01, sp_flat, sp_idx_3, sp_idx_1, valid_float, safe_counts, B, K, device
        )  # 81D

        texture_feat = self._compute_texture_features(
            images_01, sp_flat, sp_idx_1, valid_float, safe_counts, B, K, H, W, device
        )  # 35D

        shape_feat, centroids = self._compute_shape_features(
            sp_flat, sp_idx_1, sp_idx_2, valid_float, safe_counts, counts,
            slic_maps, B, K, H, W, device
        )  # 12D + centroids

        spatial_feat = self._compute_spatial_features(centroids, B, K, device)  # 4D

        # For relational features, need RGB mean and gradient magnitude mean
        rgb_flat = images_01.flatten(2).transpose(1, 2) * valid_float
        rgb_sum = torch.zeros(B, K, 3, device=device)
        rgb_sum.scatter_add_(1, sp_idx_3, rgb_flat)
        rgb_mean = rgb_sum / safe_counts

        gray = images_01.mean(dim=1, keepdim=True)
        grad_x = F.conv2d(gray, self.sobel_x, padding=1)
        grad_y = F.conv2d(gray, self.sobel_y, padding=1)
        mag = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)
        mag_flat = mag.flatten(2).transpose(1, 2) * valid_float
        mag_sum = torch.zeros(B, K, 1, device=device)
        mag_sum.scatter_add_(1, sp_idx_1, mag_flat)
        grad_mag_mean = mag_sum / safe_counts

        appear_rel, geo_rel = self._compute_relational_features(
            slic_maps, rgb_mean, grad_mag_mean, centroids, mask, B, K, H, W, device
        )  # 2D + 2D

        mask_float = mask.unsqueeze(-1).float()

        # ── Geometry path: shape(12) + spatial(4) + geo_rel(2) = 18D ──
        geometry_raw = torch.cat([shape_feat, spatial_feat, geo_rel], dim=2)  # 18D
        geometry_raw = torch.nan_to_num(geometry_raw, nan=0.0, posinf=1.0, neginf=-1.0)
        geometry_raw = geometry_raw * mask_float
        geo_embed = self.geometry_mlp(geometry_raw)  # 64D

        # ── Appearance path: color(81) + texture(35) + appear_rel(2) = 118D ──
        appearance_raw = torch.cat([color_feat, texture_feat, appear_rel], dim=2)  # 118D
        appearance_raw = torch.nan_to_num(appearance_raw, nan=0.0, posinf=1.0, neginf=-1.0)
        appearance_raw = appearance_raw * mask_float

        # ── Combined projection: 118D + 64D = 182D → hidden_dim ──
        combined = torch.cat([appearance_raw, geo_embed], dim=2)  # 182D
        features = self.input_norm(combined)
        features = self.projection(features)
        features = self.output_norm(features)

        if self.debug:
            print(f"[SuperpixelFeatureExtractorGPU] geometry_raw: {geometry_raw.shape}, "
                  f"mean={geometry_raw.mean():.4f}, std={geometry_raw.std():.4f}")
            print(f"[SuperpixelFeatureExtractorGPU] geo_embed: {geo_embed.shape}, "
                  f"mean={geo_embed.mean():.4f}, std={geo_embed.std():.4f}")
            print(f"[SuperpixelFeatureExtractorGPU] appearance_raw: {appearance_raw.shape}, "
                  f"mean={appearance_raw.mean():.4f}, std={appearance_raw.std():.4f}")
            print(f"[SuperpixelFeatureExtractorGPU] combined: {combined.shape}, "
                  f"mean={combined.mean():.4f}, std={combined.std():.4f}")
            print(f"[SuperpixelFeatureExtractorGPU] output: {features.shape}, "
                  f"mean={features.mean():.4f}, std={features.std():.4f}")

        return features, mask, centroids


# =============================================================================
# SLIC Feature Extractor (Multi-Scale Backbone wrapper for DINO)
# =============================================================================
class SLICFeatureExtractor(nn.Module):
    """SLIC Superpixel backbone for DINO (multi-scale, GPU-accelerated).

    Wraps SuperpixelFeatureExtractorGPU with multi-scale support.
    Runs the GPU feature extractor once per scale level, then concatenates
    tokens with padding to produce the output format expected by SLICTransformer.

    Output dict:
        tokens:       [bs, N_total, output_dim]  — projected superpixel features
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

        # GPU feature extractor (shared across levels)
        self.feature_extractor = SuperpixelFeatureExtractorGPU(
            hidden_dim=output_dim,
            max_superpixels=max_superpixels_per_level,
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
        """Forward pass: extract multi-scale SLIC features as token sequences.

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

            # GPU feature extraction
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
