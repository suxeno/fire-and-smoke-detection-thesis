"""
Superpixel Feature Extraction Module

Extracts rich features from superpixels:
- Color statistics RGB (mean, std, range) - 9D
- Color statistics LAB (mean, std, range) - 9D  
- Color statistics HSV (mean, std, range) - 9D
- Gradient/texture Sobel (mean, std for x,y) - 4D
- GLCM texture (contrast, energy, homogeneity, entropy) - 4D
- LBP histogram (16-bin normalized) - 16D
- HOG descriptor (8-bin normalized) - 8D
- Shape (variance, covariance, area) - 4D
- Position (centroid) - 2D
- Neighbor features (color diff, texture diff, boundary strength, degree) - 4D
Total: 69D per superpixel (single scale)

Multi-scale mode: 3 scales (150, 300, 600) × 69D = 207D

SuperpixelPool class supports two modes:
1. Standalone (DETR-SLIC): Extracts features, projects to hidden_dim
2. With CNN (DETR-Hybrid): Pools CNN features + handcrafted features, fuses both
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SuperpixelPool(nn.Module):
    """
    Unified superpixel feature extraction module with rich handcrafted features.
    
    Features per superpixel (69D single-scale, 207D multi-scale):
    - RGB color stats: mean(3) + std(3) + range(3) = 9D
    - LAB color stats: mean(3) + std(3) + range(3) = 9D
    - HSV color stats: mean(3) + std(3) + range(3) = 9D
    - Sobel gradient: mean_x, std_x, mean_y, std_y = 4D
    - GLCM texture: contrast, energy, homogeneity, entropy = 4D
    - LBP histogram: 16-bin normalized = 16D
    - HOG descriptor: 8-bin normalized = 8D
    - Shape: var_y, var_x, cov, area = 4D
    - Position: centroid_y, centroid_x = 2D
    - Neighbor: color_diff, texture_diff, boundary_strength, degree = 4D
    
    Args:
        hidden_dim: Output feature dimension (default: 256)
        max_superpixels: Maximum number of superpixels for primary scale (default: 300)
        use_cnn_features: Whether to expect and fuse CNN features (default: False)
        cnn_dim: Dimension of CNN features if use_cnn_features=True
        use_multiscale: Whether to use multi-scale superpixels (default: False)
        superpixel_scales: List of superpixel counts for multi-scale (default: [150, 300, 600])
    """
    
    def __init__(self, 
                 hidden_dim: int = 256, 
                 max_superpixels: int = 300,
                 use_cnn_features: bool = False,
                 cnn_dim: int = 256,
                 use_multiscale: bool = False,
                 superpixel_scales: list = None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_superpixels = max_superpixels
        self.use_cnn_features = use_cnn_features
        self.cnn_dim = cnn_dim
        self.use_multiscale = use_multiscale
        self.superpixel_scales = superpixel_scales or [150, 300, 600]
        
        # Feature dimensions per superpixel (single scale)
        # RGB: 9, LAB: 9, HSV: 9, Sobel: 4, GLCM: 4, LBP: 16, HOG: 8, Shape: 4, Position: 2, Neighbor: 4
        self.single_scale_dim = 69
        
        if use_multiscale:
            # Multi-scale: features from each scale concatenated
            self.superpixel_feature_dim = self.single_scale_dim * len(self.superpixel_scales)
        else:
            self.superpixel_feature_dim = self.single_scale_dim
        
        if use_cnn_features:
            # Project CNN features if dimension differs
            if cnn_dim != hidden_dim:
                self.cnn_proj = nn.Linear(cnn_dim, hidden_dim)
            else:
                self.cnn_proj = nn.Identity()
            
            # Fusion: CNN features (hidden_dim) + superpixel features -> hidden_dim
            self.fusion_mlp = nn.Sequential(
                nn.Linear(hidden_dim + self.superpixel_feature_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
        else:
            # Standalone: features -> hidden_dim (deeper network for richer representation)
            self.feature_proj = nn.Sequential(
                nn.Linear(self.superpixel_feature_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
            )
        
        # Sobel kernels for gradient computation
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
        
        # LBP offsets (8 neighbors for uniform LBP)
        # Stored as (dy, dx) pairs for 8 neighbors in circular order
        lbp_offsets = torch.tensor([
            [-1, -1], [-1, 0], [-1, 1],
            [0, 1], [1, 1], [1, 0],
            [1, -1], [0, -1]
        ], dtype=torch.long)
        self.register_buffer('lbp_offsets', lbp_offsets)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def rgb_to_lab(self, rgb):
        """
        Convert RGB to LAB color space (differentiable approximation).
        Args:
            rgb: (B, 3, H, W) in range [0, 1]
        Returns:
            lab: (B, 3, H, W) L in [0,1], a,b in [-1,1] (normalized)
        """
        # RGB to XYZ (sRGB with D65 illuminant)
        r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
        
        # Linearize sRGB
        mask = rgb > 0.04045
        rgb_linear = torch.where(mask, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
        r_lin, g_lin, b_lin = rgb_linear[:, 0:1], rgb_linear[:, 1:2], rgb_linear[:, 2:3]
        
        # RGB to XYZ matrix
        x = r_lin * 0.4124564 + g_lin * 0.3575761 + b_lin * 0.1804375
        y = r_lin * 0.2126729 + g_lin * 0.7151522 + b_lin * 0.0721750
        z = r_lin * 0.0193339 + g_lin * 0.1191920 + b_lin * 0.9503041
        
        # Normalize by D65 white point
        x = x / 0.95047
        z = z / 1.08883
        
        # XYZ to LAB
        epsilon = 0.008856
        kappa = 903.3
        
        fx = torch.where(x > epsilon, x ** (1/3), (kappa * x + 16) / 116)
        fy = torch.where(y > epsilon, y ** (1/3), (kappa * y + 16) / 116)
        fz = torch.where(z > epsilon, z ** (1/3), (kappa * z + 16) / 116)
        
        L = 116 * fy - 16  # L in [0, 100]
        a = 500 * (fx - fy)  # a in [-128, 127]
        b_lab = 200 * (fy - fz)  # b in [-128, 127]
        
        # Normalize to [0,1] for L and [-1,1] for a,b
        L = L / 100.0
        a = a / 128.0
        b_lab = b_lab / 128.0
        
        return torch.cat([L, a, b_lab], dim=1)
    
    def rgb_to_hsv(self, rgb):
        """
        Convert RGB to HSV color space (differentiable).
        Args:
            rgb: (B, 3, H, W) in range [0, 1]
        Returns:
            hsv: (B, 3, H, W) all in [0, 1]
        """
        r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
        
        max_rgb, _ = rgb.max(dim=1, keepdim=True)
        min_rgb, _ = rgb.min(dim=1, keepdim=True)
        diff = max_rgb - min_rgb + 1e-8
        
        # Value
        v = max_rgb
        
        # Saturation
        s = torch.where(max_rgb > 1e-8, diff / (max_rgb + 1e-8), torch.zeros_like(max_rgb))
        
        # Hue
        h = torch.zeros_like(max_rgb)
        
        # When max is R
        mask_r = (max_rgb == r)
        h = torch.where(mask_r, ((g - b) / diff) % 6, h)
        
        # When max is G
        mask_g = (max_rgb == g)
        h = torch.where(mask_g, (b - r) / diff + 2, h)
        
        # When max is B
        mask_b = (max_rgb == b)
        h = torch.where(mask_b, (r - g) / diff + 4, h)
        
        h = h / 6.0  # Normalize to [0, 1]
        h = torch.clamp(h, 0, 1)
        
        return torch.cat([h, s, v], dim=1)
    
    def compute_lbp(self, gray):
        """
        Compute Local Binary Pattern codes (simplified uniform LBP).
        Args:
            gray: (B, 1, H, W) grayscale image
        Returns:
            lbp: (B, 1, H, W) LBP codes (0-255 mapped to 0-15 bins)
        """
        B, _, H, W = gray.shape
        device = gray.device
        
        # Pad image for neighbor access
        gray_pad = F.pad(gray, (1, 1, 1, 1), mode='replicate')
        
        # Compute LBP code by comparing with 8 neighbors
        lbp_code = torch.zeros(B, 1, H, W, device=device)
        
        for i, (dy, dx) in enumerate(self.lbp_offsets):
            neighbor = gray_pad[:, :, 1+dy:H+1+dy, 1+dx:W+1+dx]
            bit = (neighbor >= gray).float()
            lbp_code = lbp_code + bit * (2 ** i)
        
        # Map 256 possible codes to 16 bins (simplified uniform LBP)
        lbp_bin = (lbp_code / 16.0).floor().clamp(0, 15)
        
        return lbp_bin
    
    def compute_hog_simple(self, gray, grad_x, grad_y):
        """
        Compute simplified HOG-like orientation histogram.
        Args:
            gray: (B, 1, H, W) grayscale
            grad_x: (B, 1, H, W) horizontal gradient
            grad_y: (B, 1, H, W) vertical gradient
        Returns:
            orientation: (B, 1, H, W) quantized orientation bin (0-7)
            magnitude: (B, 1, H, W) gradient magnitude
        """
        magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)
        orientation = torch.atan2(grad_y, grad_x)  # [-pi, pi]
        
        # Map to [0, 8) bins (unsigned orientation, 0-180 degrees)
        orientation = (orientation + math.pi) / (2 * math.pi) * 8  # [0, 8]
        orientation = orientation.floor().clamp(0, 7)
        
        return orientation, magnitude
    
    def compute_glcm_features(self, gray, sp_flat, sp_idx, valid_float, safe_counts, B, K, device):
        """
        Compute GLCM-inspired texture features (differentiable approximation).
        Instead of building full co-occurrence matrices, we compute statistics
        that capture similar texture properties.
        
        Returns:
            glcm_features: (B, K, 4) - contrast, energy, homogeneity, entropy proxy
        """
        H_W = sp_flat.shape[1]
        
        # Quantize gray to 16 levels for efficiency
        gray_flat = (gray.flatten(2).transpose(1, 2) * 15).clamp(0, 15)  # (B, H*W, 1)
        gray_flat = gray_flat * valid_float
        
        # === Contrast: variance of intensity within superpixel ===
        gray_sum = torch.zeros(B, K, 1, device=device)
        gray_sum.scatter_add_(1, sp_idx, gray_flat)
        gray_mean = gray_sum / safe_counts
        
        gray_sq_sum = torch.zeros(B, K, 1, device=device)
        gray_sq_sum.scatter_add_(1, sp_idx, gray_flat ** 2)
        contrast = (gray_sq_sum / safe_counts - gray_mean ** 2).clamp(min=0)
        contrast = contrast / 225.0  # Normalize (max variance for 0-15 range)
        
        # === Energy: inverse of spread (concentrated = high energy) ===
        # Approximate by 1 / (1 + variance)
        energy = 1.0 / (1.0 + contrast * 225.0)
        
        # === Homogeneity: how uniform the region is ===
        # Approximate by 1 / (1 + std)
        gray_std = torch.sqrt(contrast * 225.0 + 1e-6)
        homogeneity = 1.0 / (1.0 + gray_std / 15.0)
        
        # === Entropy proxy: based on range of values ===
        # Higher range = more disorder = higher entropy
        # Use std as proxy (normalized)
        entropy = gray_std / 15.0
        
        return torch.cat([contrast, energy, homogeneity, entropy], dim=2)
    
    def compute_neighbor_features(self, sp_map, color_mean, grad_magnitude_mean, centroids, mask, B, K, H, W, device):
        """
        Compute neighbor/adjacency features for each superpixel.
        
        Returns:
            neighbor_features: (B, K, 4) - color_diff, texture_diff, boundary_strength, degree
        """
        # Build adjacency by checking horizontal and vertical neighbors
        # This is an efficient approximation
        
        # Shift superpixel map to find neighbors
        sp_map_pad = F.pad(sp_map.unsqueeze(1).float(), (1, 1, 1, 1), mode='replicate').squeeze(1).long()
        
        # Get neighbor superpixel IDs (right, down, left, up)
        neighbors_r = sp_map_pad[:, 1:-1, 2:]  # right
        neighbors_d = sp_map_pad[:, 2:, 1:-1]  # down
        neighbors_l = sp_map_pad[:, 1:-1, :-2]  # left
        neighbors_u = sp_map_pad[:, :-2, 1:-1]  # up
        
        # Find boundary pixels (where current != neighbor)
        boundary_r = (sp_map != neighbors_r) & (sp_map >= 0) & (neighbors_r >= 0)
        boundary_d = (sp_map != neighbors_d) & (sp_map >= 0) & (neighbors_d >= 0)
        
        # Initialize neighbor features
        neighbor_color_diff = torch.zeros(B, K, 1, device=device)
        neighbor_texture_diff = torch.zeros(B, K, 1, device=device)
        neighbor_count = torch.zeros(B, K, 1, device=device)
        
        # For each superpixel, find its neighbors and compute differences
        # This is done efficiently using scatter operations
        
        # Flatten boundary info
        boundary_mask = (boundary_r | boundary_d).flatten(1)  # (B, H*W)
        sp_flat = sp_map.flatten(1)
        
        # Get neighbor IDs at boundary pixels
        neighbor_ids_r = neighbors_r.flatten(1)
        neighbor_ids_d = neighbors_d.flatten(1)
        
        # For boundary pixels, accumulate neighbor relationships
        for b in range(B):
            # Find unique (superpixel, neighbor) pairs
            bnd_pixels = boundary_mask[b].nonzero(as_tuple=True)[0]
            if len(bnd_pixels) == 0:
                continue
            
            sp_ids = sp_flat[b, bnd_pixels]
            neigh_r = neighbor_ids_r[b, bnd_pixels]
            neigh_d = neighbor_ids_d[b, bnd_pixels]
            
            # Compute color difference for right neighbors
            valid_r = (sp_ids != neigh_r) & (sp_ids >= 0) & (sp_ids < K) & (neigh_r >= 0) & (neigh_r < K)
            if valid_r.any():
                sp_valid = sp_ids[valid_r]
                neigh_valid = neigh_r[valid_r]
                color_diff = (color_mean[b, sp_valid] - color_mean[b, neigh_valid]).abs().mean(dim=1, keepdim=True)
                texture_diff = (grad_magnitude_mean[b, sp_valid] - grad_magnitude_mean[b, neigh_valid]).abs()
                
                neighbor_color_diff[b].scatter_add_(0, sp_valid.unsqueeze(1), color_diff)
                neighbor_texture_diff[b].scatter_add_(0, sp_valid.unsqueeze(1), texture_diff)
                neighbor_count[b].scatter_add_(0, sp_valid.unsqueeze(1), torch.ones_like(color_diff))
        
        # Average by neighbor count
        safe_neighbor_count = neighbor_count.clamp(min=1)
        mean_color_diff = neighbor_color_diff / safe_neighbor_count
        mean_texture_diff = neighbor_texture_diff / safe_neighbor_count
        
        # Boundary strength: use gradient magnitude at boundaries (already computed)
        boundary_strength = mean_texture_diff
        
        # Degree: number of neighbors (normalized by max possible ~4)
        degree = neighbor_count / 4.0
        
        return torch.cat([mean_color_diff, mean_texture_diff, boundary_strength, degree], dim=2)
    
    def extract_superpixel_features(self, images, superpixel_map):
        """
        Extract rich 69D features from each superpixel.
        Uses memory-efficient scatter_add operations.
        
        Args:
            images: (B, 3, H, W) normalized RGB images
            superpixel_map: (B, H, W) superpixel indices (-1 for padding)
            
        Returns:
            features: (B, K, 69) raw superpixel features
            mask: (B, K) True for valid superpixels
            centroids: (B, K, 2) normalized (y, x) coordinates
        """
        B, C, H, W = images.shape
        K = self.max_superpixels
        device = images.device
        
        # Resize superpixel map to image size if needed
        H_sp, W_sp = superpixel_map.shape[1:]
        if H_sp != H or W_sp != W:
            sp_map = F.interpolate(
                superpixel_map.unsqueeze(1).float(),
                size=(H, W),
                mode='nearest'
            ).squeeze(1).long()
        else:
            sp_map = superpixel_map
        
        # Flatten for scatter operations
        img_flat = images.flatten(2).transpose(1, 2)  # (B, H*W, 3)
        sp_flat = sp_map.flatten(1)  # (B, H*W)
        
        # Valid pixels mask (ignore padding -1)
        valid_pixels = (sp_flat >= 0) & (sp_flat < K)
        sp_flat_safe = sp_flat.clone()
        sp_flat_safe[~valid_pixels] = 0
        
        # Index tensors for scatter_add
        sp_idx = sp_flat_safe.unsqueeze(-1)  # (B, H*W, 1)
        sp_idx_3 = sp_idx.expand(-1, -1, 3)  # (B, H*W, 3)
        sp_idx_2 = sp_idx.expand(-1, -1, 2)  # (B, H*W, 2)
        
        # Pixel validity mask for masking values
        valid_float = valid_pixels.unsqueeze(-1).float()  # (B, H*W, 1)
        
        # === COUNTS ===
        ones = valid_float
        counts = torch.zeros(B, K, 1, device=device)
        counts.scatter_add_(1, sp_idx, ones)
        safe_counts = counts.clamp(min=1)
        mask = (counts.squeeze(2) > 0)  # (B, K)
        
        # ==================== COLOR FEATURES (27D) ====================
        
        # --- RGB Color Statistics (9D) ---
        masked_rgb = img_flat * valid_float
        
        rgb_sum = torch.zeros(B, K, 3, device=device)
        rgb_sum.scatter_add_(1, sp_idx_3, masked_rgb)
        rgb_mean = rgb_sum / safe_counts
        
        rgb_sq_sum = torch.zeros(B, K, 3, device=device)
        rgb_sq_sum.scatter_add_(1, sp_idx_3, masked_rgb ** 2)
        rgb_var = (rgb_sq_sum / safe_counts - rgb_mean ** 2).clamp(min=0)
        rgb_std = torch.sqrt(rgb_var + 1e-6)
        rgb_range = 2 * rgb_std  # Approximation
        
        # --- LAB Color Statistics (9D) ---
        lab = self.rgb_to_lab(images)  # (B, 3, H, W)
        lab_flat = lab.flatten(2).transpose(1, 2) * valid_float
        
        lab_sum = torch.zeros(B, K, 3, device=device)
        lab_sum.scatter_add_(1, sp_idx_3, lab_flat)
        lab_mean = lab_sum / safe_counts
        
        lab_sq_sum = torch.zeros(B, K, 3, device=device)
        lab_sq_sum.scatter_add_(1, sp_idx_3, lab_flat ** 2)
        lab_var = (lab_sq_sum / safe_counts - lab_mean ** 2).clamp(min=0)
        lab_std = torch.sqrt(lab_var + 1e-6)
        lab_range = 2 * lab_std
        
        # --- HSV Color Statistics (9D) ---
        hsv = self.rgb_to_hsv(images)  # (B, 3, H, W)
        hsv_flat = hsv.flatten(2).transpose(1, 2) * valid_float
        
        hsv_sum = torch.zeros(B, K, 3, device=device)
        hsv_sum.scatter_add_(1, sp_idx_3, hsv_flat)
        hsv_mean = hsv_sum / safe_counts
        
        hsv_sq_sum = torch.zeros(B, K, 3, device=device)
        hsv_sq_sum.scatter_add_(1, sp_idx_3, hsv_flat ** 2)
        hsv_var = (hsv_sq_sum / safe_counts - hsv_mean ** 2).clamp(min=0)
        hsv_std = torch.sqrt(hsv_var + 1e-6)
        hsv_range = 2 * hsv_std
        
        # ==================== GRADIENT/TEXTURE FEATURES (32D) ====================
        
        gray = images.mean(dim=1, keepdim=True)  # (B, 1, H, W)
        grad_x = F.conv2d(gray, self.sobel_x, padding=1)
        grad_y = F.conv2d(gray, self.sobel_y, padding=1)
        
        # --- Sobel Gradient Stats (4D) ---
        grad_x_flat = grad_x.flatten(2).transpose(1, 2) * valid_float
        grad_y_flat = grad_y.flatten(2).transpose(1, 2) * valid_float
        
        grad_x_sum = torch.zeros(B, K, 1, device=device)
        grad_x_sum.scatter_add_(1, sp_idx, grad_x_flat)
        grad_x_mean = grad_x_sum / safe_counts
        
        grad_x_sq_sum = torch.zeros(B, K, 1, device=device)
        grad_x_sq_sum.scatter_add_(1, sp_idx, grad_x_flat ** 2)
        grad_x_var = (grad_x_sq_sum / safe_counts - grad_x_mean ** 2).clamp(min=0)
        grad_x_std = torch.sqrt(grad_x_var + 1e-6)
        
        grad_y_sum = torch.zeros(B, K, 1, device=device)
        grad_y_sum.scatter_add_(1, sp_idx, grad_y_flat)
        grad_y_mean = grad_y_sum / safe_counts
        
        grad_y_sq_sum = torch.zeros(B, K, 1, device=device)
        grad_y_sq_sum.scatter_add_(1, sp_idx, grad_y_flat ** 2)
        grad_y_var = (grad_y_sq_sum / safe_counts - grad_y_mean ** 2).clamp(min=0)
        grad_y_std = torch.sqrt(grad_y_var + 1e-6)
        
        # Gradient magnitude for neighbor features
        grad_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)
        grad_mag_flat = grad_magnitude.flatten(2).transpose(1, 2) * valid_float
        grad_mag_sum = torch.zeros(B, K, 1, device=device)
        grad_mag_sum.scatter_add_(1, sp_idx, grad_mag_flat)
        grad_mag_mean = grad_mag_sum / safe_counts
        
        gradient_features = torch.cat([grad_x_mean, grad_x_std, grad_y_mean, grad_y_std], dim=2)  # 4D
        
        # --- GLCM Texture Features (4D) ---
        glcm_features = self.compute_glcm_features(gray, sp_flat, sp_idx, valid_float, safe_counts, B, K, device)
        
        # --- LBP Histogram (16D) ---
        lbp_codes = self.compute_lbp(gray)  # (B, 1, H, W) with values 0-15
        lbp_flat = lbp_codes.flatten(2).transpose(1, 2)  # (B, H*W, 1)
        
        # Build 16-bin histogram per superpixel
        lbp_hist = torch.zeros(B, K, 16, device=device)
        for bin_idx in range(16):
            bin_mask = ((lbp_flat >= bin_idx) & (lbp_flat < bin_idx + 1)).float() * valid_float
            lbp_hist[:, :, bin_idx:bin_idx+1].scatter_add_(1, sp_idx, bin_mask)
        
        # Normalize histogram
        lbp_hist = lbp_hist / safe_counts.clamp(min=1)
        
        # --- HOG Descriptor (8D) ---
        orientation, magnitude = self.compute_hog_simple(gray, grad_x, grad_y)
        orient_flat = orientation.flatten(2).transpose(1, 2)  # (B, H*W, 1)
        mag_flat = magnitude.flatten(2).transpose(1, 2) * valid_float  # (B, H*W, 1)
        
        # Build 8-bin weighted histogram per superpixel
        hog_hist = torch.zeros(B, K, 8, device=device)
        for bin_idx in range(8):
            bin_mask = ((orient_flat >= bin_idx) & (orient_flat < bin_idx + 1)).float()
            weighted_bin = bin_mask * mag_flat
            hog_hist[:, :, bin_idx:bin_idx+1].scatter_add_(1, sp_idx, weighted_bin)
        
        # Normalize histogram
        hog_sum = hog_hist.sum(dim=2, keepdim=True).clamp(min=1e-6)
        hog_hist = hog_hist / hog_sum
        
        # ==================== SHAPE & POSITION (6D) ====================
        
        y_coords = torch.linspace(0, 1, H, device=device).view(1, H, 1).expand(B, H, W).flatten(1)
        x_coords = torch.linspace(0, 1, W, device=device).view(1, 1, W).expand(B, H, W).flatten(1)
        coords = torch.stack([y_coords, x_coords], dim=2) * valid_float  # (B, H*W, 2)
        
        # Centroids (2D)
        coord_sum = torch.zeros(B, K, 2, device=device)
        coord_sum.scatter_add_(1, sp_idx_2, coords)
        centroids = coord_sum / safe_counts
        
        # Second moments for shape (4D)
        coords_sq = (torch.stack([y_coords, x_coords], dim=2) ** 2) * valid_float
        coords_xy = (y_coords * x_coords).unsqueeze(-1) * valid_float
        
        mean_sq_sum = torch.zeros(B, K, 2, device=device)
        mean_sq_sum.scatter_add_(1, sp_idx_2, coords_sq)
        mean_sq = mean_sq_sum / safe_counts
        
        mean_xy_sum = torch.zeros(B, K, 1, device=device)
        mean_xy_sum.scatter_add_(1, sp_idx, coords_xy)
        mean_xy = mean_xy_sum / safe_counts
        
        var_y = (mean_sq[:, :, 0:1] - centroids[:, :, 0:1] ** 2).clamp(min=0)
        var_x = (mean_sq[:, :, 1:2] - centroids[:, :, 1:2] ** 2).clamp(min=0)
        cov_yx = mean_xy - centroids[:, :, 0:1] * centroids[:, :, 1:2]
        area = counts / (H * W)
        
        shape_features = torch.cat([var_y, var_x, cov_yx, area], dim=2)  # 4D
        
        # ==================== NEIGHBOR FEATURES (4D) ====================
        
        neighbor_features = self.compute_neighbor_features(
            sp_map, rgb_mean, grad_mag_mean, centroids, mask, B, K, H, W, device
        )
        
        # ==================== COMBINE ALL FEATURES (69D) ====================
        raw_features = torch.cat([
            rgb_mean,           # 3D
            rgb_std,            # 3D
            rgb_range,          # 3D
            lab_mean,           # 3D
            lab_std,            # 3D
            lab_range,          # 3D
            hsv_mean,           # 3D
            hsv_std,            # 3D
            hsv_range,          # 3D
            gradient_features,  # 4D
            glcm_features,      # 4D
            lbp_hist,           # 16D
            hog_hist,           # 8D
            shape_features,     # 4D
            centroids,          # 2D
            neighbor_features,  # 4D
        ], dim=2)  # Total: 69D
        
        return raw_features, mask, centroids
    
    def pool_cnn_features(self, cnn_features, superpixel_map):
        """
        Pool CNN features within each superpixel using mean pooling.
        Uses memory-efficient scatter_add.
        
        Args:
            cnn_features: (B, C, H_feat, W_feat) CNN feature map
            superpixel_map: (B, H, W) superpixel indices
            
        Returns:
            pooled: (B, K, C) pooled features per superpixel
        """
        B, C, H_feat, W_feat = cnn_features.shape
        K = self.max_superpixels
        device = cnn_features.device
        
        # Resize superpixel map to feature map size
        _, H, W = superpixel_map.shape
        if H != H_feat or W != W_feat:
            sp_map = F.interpolate(
                superpixel_map.unsqueeze(1).float(),
                size=(H_feat, W_feat),
                mode='nearest'
            ).squeeze(1).long()
        else:
            sp_map = superpixel_map
        
        # Flatten
        feat_flat = cnn_features.flatten(2).transpose(1, 2)  # (B, H*W, C)
        sp_flat = sp_map.flatten(1)  # (B, H*W)
        
        valid_pixels = (sp_flat >= 0) & (sp_flat < K)
        sp_flat_safe = sp_flat.clone()
        sp_flat_safe[~valid_pixels] = 0
        
        sp_idx = sp_flat_safe.unsqueeze(-1).expand(-1, -1, C)  # (B, H*W, C)
        valid_float = valid_pixels.unsqueeze(-1).float()
        
        # Counts
        ones = valid_float
        counts = torch.zeros(B, K, 1, device=device)
        counts.scatter_add_(1, sp_flat_safe.unsqueeze(-1), ones)
        safe_counts = counts.clamp(min=1)
        
        # Pool features
        feat_sum = torch.zeros(B, K, C, device=device)
        feat_sum.scatter_add_(1, sp_idx, feat_flat * valid_float)
        pooled = feat_sum / safe_counts
        
        return pooled
    
    def extract_multiscale_features(self, images, superpixel_maps):
        """
        Extract features from multiple superpixel scales.
        
        Args:
            images: (B, 3, H, W) normalized images
            superpixel_maps: dict with keys like 'slic_150', 'slic_300', 'slic_600'
                            or list of (B, H, W) superpixel maps
            
        Returns:
            features: (B, K, 69*num_scales) concatenated features from all scales
            mask: (B, K) True for valid superpixels (from primary scale)
            centroids: (B, K, 2) from primary scale
        """
        all_features = []
        primary_mask = None
        primary_centroids = None
        
        for i, scale in enumerate(self.superpixel_scales):
            # Get superpixel map for this scale
            if isinstance(superpixel_maps, dict):
                sp_map = superpixel_maps.get(f'slic_{scale}')
            elif isinstance(superpixel_maps, (list, tuple)):
                sp_map = superpixel_maps[i] if i < len(superpixel_maps) else superpixel_maps[-1]
            else:
                sp_map = superpixel_maps
            
            if sp_map is None:
                # Fallback to primary map if scale not available
                sp_map = superpixel_maps if not isinstance(superpixel_maps, (dict, list, tuple)) else list(superpixel_maps.values())[0]
            
            # Temporarily set max_superpixels for this scale
            original_max = self.max_superpixels
            self.max_superpixels = scale
            
            features, mask, centroids = self.extract_superpixel_features(images, sp_map)
            
            self.max_superpixels = original_max
            
            # Store primary scale mask and centroids
            if scale == self.max_superpixels or i == 1:  # Default to 300 (middle scale)
                primary_mask = mask
                primary_centroids = centroids
            
            # Pad or truncate to primary scale size for concatenation
            B, K_scale, D = features.shape
            K_primary = self.max_superpixels
            
            if K_scale < K_primary:
                # Pad with zeros
                padding = torch.zeros(B, K_primary - K_scale, D, device=features.device)
                features = torch.cat([features, padding], dim=1)
            elif K_scale > K_primary:
                # Truncate (use first K_primary superpixels)
                features = features[:, :K_primary, :]
            
            all_features.append(features)
        
        # Concatenate features from all scales
        combined_features = torch.cat(all_features, dim=2)  # (B, K, 69*num_scales)
        
        return combined_features, primary_mask, primary_centroids
    
    def forward(self, images, superpixel_map, cnn_features=None, multiscale_maps=None):
        """
        Extract superpixel features, optionally fusing with CNN features.
        
        For DETR-SLIC (use_cnn_features=False):
            features, mask, centroids = pool(images, sp_map)
        
        For DETR-Hybrid (use_cnn_features=True):
            features, mask, centroids = pool(images, sp_map, cnn_features=backbone_out)
        
        For Multi-scale mode:
            features, mask, centroids = pool(images, sp_map, multiscale_maps={'slic_150': ..., ...})
        
        Args:
            images: (B, 3, H, W) normalized images
            superpixel_map: (B, H, W) superpixel indices (primary scale)
            cnn_features: Optional (B, C, H_feat, W_feat) CNN features for hybrid mode
            multiscale_maps: Optional dict with multi-scale superpixel maps
            
        Returns:
            features: (B, K, hidden_dim) superpixel features
            mask: (B, K) True for valid superpixels
            centroids: (B, K, 2) normalized centroids
        """
        # Extract superpixel features (single or multi-scale)
        if self.use_multiscale and multiscale_maps is not None:
            sp_features, mask, centroids = self.extract_multiscale_features(images, multiscale_maps)
        else:
            sp_features, mask, centroids = self.extract_superpixel_features(images, superpixel_map)
        
        if self.use_cnn_features:
            if cnn_features is None:
                raise ValueError("use_cnn_features=True but cnn_features not provided")
            
            # Pool and project CNN features
            cnn_pooled = self.pool_cnn_features(cnn_features, superpixel_map)
            cnn_pooled = self.cnn_proj(cnn_pooled)
            
            # Fuse CNN + superpixel features
            combined = torch.cat([cnn_pooled, sp_features], dim=2)
            features = self.fusion_mlp(combined)
        else:
            # Project superpixel features only
            features = self.feature_proj(sp_features)
        
        # Zero out invalid superpixels
        features = features * mask.unsqueeze(-1).float()
        
        return features, mask, centroids
