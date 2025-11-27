import torch
import torch.nn as nn
import torch.nn.functional as F


class FourierShapeDescriptor(nn.Module):
    """
    Computes Fourier Descriptors for superpixel boundary shapes.
    
    For each superpixel, extracts boundary pixels, converts to complex sequence,
    applies FFT, and keeps low-frequency coefficients as shape descriptor.
    
    Fourier descriptors are invariant to:
    - Translation (by centering on centroid)
    - Scale (by normalizing by DC component)
    - Rotation (by using magnitudes only)
    """
    
    def __init__(self, n_coeffs=16, use_magnitude_only=True):
        """
        Args:
            n_coeffs: Number of Fourier coefficients to keep (low frequencies)
            use_magnitude_only: If True, use only magnitudes (rotation invariant)
                               If False, use real+imag parts (preserves orientation)
        """
        super().__init__()
        self.n_coeffs = n_coeffs
        self.use_magnitude_only = use_magnitude_only
        # Output dimension: n_coeffs if magnitude only, else 2*n_coeffs (real+imag)
        self.out_dim = n_coeffs if use_magnitude_only else n_coeffs * 2
    
    def extract_boundary_mask(self, sp_mask):
        """
        Extract boundary pixels from a binary superpixel mask.
        Boundary = pixels that have at least one neighbor outside the superpixel.
        
        Args:
            sp_mask: (H, W) binary mask of one superpixel
        Returns:
            boundary_mask: (H, W) binary mask of boundary pixels
        """
        # Pad to handle edges
        padded = F.pad(sp_mask.unsqueeze(0).unsqueeze(0).float(), (1, 1, 1, 1), mode='constant', value=0)
        
        # Check 4-connected neighbors
        center = padded[:, :, 1:-1, 1:-1]
        up = padded[:, :, :-2, 1:-1]
        down = padded[:, :, 2:, 1:-1]
        left = padded[:, :, 1:-1, :-2]
        right = padded[:, :, 1:-1, 2:]
        
        # Boundary = inside superpixel AND has at least one neighbor outside
        neighbors_inside = up + down + left + right
        boundary = (center > 0) & (neighbors_inside < 4)
        
        return boundary.squeeze(0).squeeze(0)
    
    def order_boundary_points(self, boundary_coords, max_points=64):
        """
        Order boundary points to form a closed contour.
        Uses a simple greedy nearest-neighbor approach.
        
        Args:
            boundary_coords: (N, 2) tensor of (y, x) coordinates
            max_points: Maximum number of points to sample
        Returns:
            ordered_coords: (M, 2) ordered contour points
        """
        if len(boundary_coords) <= 2:
            return boundary_coords
        
        N = len(boundary_coords)
        
        # Subsample if too many points
        if N > max_points:
            indices = torch.linspace(0, N - 1, max_points).long()
            boundary_coords = boundary_coords[indices]
            N = max_points
        
        # Start from the topmost-leftmost point
        start_idx = 0
        min_val = boundary_coords[0, 0] * 10000 + boundary_coords[0, 1]
        for i in range(1, N):
            val = boundary_coords[i, 0] * 10000 + boundary_coords[i, 1]
            if val < min_val:
                min_val = val
                start_idx = i
        
        # Greedy nearest neighbor ordering
        ordered = [start_idx]
        remaining = set(range(N)) - {start_idx}
        
        current = start_idx
        while remaining:
            current_coord = boundary_coords[current]
            # Find nearest unvisited point
            min_dist = float('inf')
            nearest = None
            for idx in remaining:
                dist = ((boundary_coords[idx] - current_coord) ** 2).sum()
                if dist < min_dist:
                    min_dist = dist
                    nearest = idx
            if nearest is not None:
                ordered.append(nearest)
                remaining.remove(nearest)
                current = nearest
            else:
                break
        
        return boundary_coords[ordered]
    
    def compute_fourier_descriptor(self, contour, centroid):
        """
        Compute Fourier descriptor from ordered contour points.
        
        Args:
            contour: (M, 2) ordered contour points (y, x)
            centroid: (2,) centroid (y, x)
        Returns:
            descriptor: (n_coeffs,) or (2*n_coeffs,) Fourier descriptor
        """
        device = contour.device
        M = len(contour)
        
        if M < 4:
            # Too few points, return zeros
            return torch.zeros(self.out_dim, device=device)
        
        # Center contour on centroid (translation invariance)
        centered = contour - centroid.unsqueeze(0)
        
        # Convert to complex: z = x + i*y (note: using x as real, y as imaginary)
        z = torch.complex(centered[:, 1], centered[:, 0])
        
        # Compute FFT
        fft_coeffs = torch.fft.fft(z)
        
        # Take first n_coeffs (low frequencies) - skip DC (index 0) for scale invariance
        # We use indices 1 to n_coeffs+1
        n_take = min(self.n_coeffs + 1, M)
        coeffs = fft_coeffs[1:n_take]
        
        # Pad if not enough coefficients
        if len(coeffs) < self.n_coeffs:
            padding = torch.zeros(self.n_coeffs - len(coeffs), dtype=torch.complex64, device=device)
            coeffs = torch.cat([coeffs, padding])
        else:
            coeffs = coeffs[:self.n_coeffs]
        
        # Normalize by the magnitude of first coefficient (scale invariance)
        scale = torch.abs(coeffs[0]) + 1e-8
        coeffs = coeffs / scale
        
        if self.use_magnitude_only:
            # Return magnitudes only (rotation invariant)
            descriptor = torch.abs(coeffs)
        else:
            # Return real and imaginary parts (preserves orientation info)
            descriptor = torch.cat([coeffs.real, coeffs.imag])
        
        return descriptor
    
    def forward(self, superpixel_map, centroids, H, W):
        """
        Compute Fourier shape descriptors for all superpixels.
        
        Args:
            superpixel_map: (B, H_sp, W_sp) superpixel indices
            centroids: (B, K, 2) precomputed centroids (y, x) normalized [0,1]
            H, W: Feature map dimensions (for coordinate scaling)
        Returns:
            fourier_desc: (B, K, out_dim) Fourier descriptors per superpixel
        """
        B, H_sp, W_sp = superpixel_map.shape
        K = centroids.shape[1]
        device = superpixel_map.device
        
        # Resize superpixel map to feature map size if different
        if H_sp != H or W_sp != W:
            sp_map = F.interpolate(
                superpixel_map.unsqueeze(1).float(), 
                size=(H, W), 
                mode='nearest'
            ).squeeze(1).long()
        else:
            sp_map = superpixel_map
        
        # Initialize output
        fourier_desc = torch.zeros(B, K, self.out_dim, device=device)
        
        # Process each batch and superpixel
        for b in range(B):
            for k in range(K):
                # Get mask for this superpixel
                sp_mask = (sp_map[b] == k)
                
                if sp_mask.sum() < 4:
                    # Too small, skip (descriptor stays zero)
                    continue
                
                # Extract boundary
                boundary_mask = self.extract_boundary_mask(sp_mask)
                
                if boundary_mask.sum() < 4:
                    continue
                
                # Get boundary coordinates
                boundary_coords = torch.stack(torch.where(boundary_mask), dim=1).float()  # (N, 2) as (y, x)
                
                # Normalize coordinates to [0, 1]
                boundary_coords[:, 0] = boundary_coords[:, 0] / H
                boundary_coords[:, 1] = boundary_coords[:, 1] / W
                
                # Order boundary points
                ordered_contour = self.order_boundary_points(boundary_coords)
                
                # Get centroid (already normalized)
                centroid = centroids[b, k]
                
                # Compute Fourier descriptor
                fourier_desc[b, k] = self.compute_fourier_descriptor(ordered_contour, centroid)
        
        return fourier_desc


class SuperpixelPool(nn.Module):
    def __init__(self, in_dim, out_dim, max_superpixels=100, pooling_type='mean',
                 use_fourier_shape=True, n_fourier_coeffs=16):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.max_superpixels = max_superpixels
        self.pooling_type = pooling_type
        self.use_fourier_shape = use_fourier_shape
        
        if in_dim != out_dim:
            self.proj = nn.Linear(in_dim, out_dim)
        else:
            self.proj = nn.Identity()
        
        # Fourier shape descriptor
        if use_fourier_shape:
            self.fourier_shape = FourierShapeDescriptor(
                n_coeffs=n_fourier_coeffs, 
                use_magnitude_only=True  # Rotation invariant
            )
            fourier_dim = self.fourier_shape.out_dim
        else:
            self.fourier_shape = None
            fourier_dim = 0
            
        # Fusion MLP for SuperFormer-style features
        # CNN features (out_dim) + Color Mean (3) + Color Std (3) + Covariance Shape (3) + Centroid (2) + Fourier (n_coeffs)
        extra_dim = 11 + fourier_dim  # 11 = 3+3+3+2
        self.fusion_mlp = nn.Sequential(
            nn.Linear(out_dim + extra_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )

    def forward(self, x, superpixel_map, images=None):
        """
        Args:
            x: Feature map (B, C, H, W)
            superpixel_map: Superpixel indices (B, H, W)
            images: Optional original images (B, 3, H_img, W_img) for color stats
        Returns:
            pooled_features: (B, K, C)
            mask: (B, K) - True if superpixel is valid (has pixels)
            centroids: (B, K, 2) - (y, x) normalized coordinates
        """
        B, C, H, W = x.shape
        K = self.max_superpixels
        
        # Flatten features and superpixel map
        x_flat = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
        sp_flat = superpixel_map.flatten(1)    # (B, H*W)
        
        # Create a mask for valid pixels (ignoring padding -1)
        valid_pixels = (sp_flat >= 0)
        
        # Handle indices for one_hot
        # 1. Replace -1 with 0 to avoid index error (we will mask these out later)
        # 2. Clamp to K-1 to handle SLIC overflow
        sp_flat_safe = sp_flat.clone()
        sp_flat_safe[~valid_pixels] = 0
        sp_flat_safe = sp_flat_safe.clamp(max=K - 1)
        
        # One-hot encoding of superpixel assignments
        # (B, H*W, K)
        one_hot = F.one_hot(sp_flat_safe, num_classes=K).float()
        
        # Mask out invalid pixels (padding) from one_hot
        # valid_pixels: (B, H*W) -> (B, H*W, 1)
        one_hot = one_hot * valid_pixels.unsqueeze(-1).float()
        
        # Calculate counts for each superpixel
        counts = one_hot.sum(dim=1)  # (B, K)
        counts = counts.unsqueeze(2) # (B, K, 1)
        safe_counts = counts + 1e-6
        
        # Pool features
        # (B, K, C) = (B, K, H*W) @ (B, H*W, C)
        # We transpose one_hot to (B, K, H*W)
        pooled_features = torch.bmm(one_hot.transpose(1, 2), x_flat)
        
        if self.pooling_type == 'mean':
            pooled_features = pooled_features / safe_counts
        
        # Project CNN features
        pooled_features = self.proj(pooled_features)
        
        # Create mask (True if superpixel exists/has pixels)
        mask = (counts.squeeze(2) > 0) # (B, K)
        
        # --- Compute Centroids and Shape Descriptors ---
        # Create grid of coordinates
        y_grid, x_grid = torch.meshgrid(torch.arange(H, device=x.device), torch.arange(W, device=x.device), indexing='ij')
        
        # Normalize coordinates to [0, 1]
        y_grid = y_grid.float() / H
        x_grid = x_grid.float() / W
        
        coords = torch.stack([y_grid, x_grid], dim=-1).flatten(0, 1) # (H*W, 2)
        coords = coords.unsqueeze(0).expand(B, -1, -1) # (B, H*W, 2)
        
        # 1. Centroids (Mean)
        centroids = torch.bmm(one_hot.transpose(1, 2), coords) / safe_counts # (B, K, 2)
        
        # 2. Shape (Covariance)
        # E[x^2], E[y^2], E[xy]
        coords_sq = coords ** 2
        coords_xy = (coords[:, :, 0] * coords[:, :, 1]).unsqueeze(2) # (B, H*W, 1)
        
        mean_sq = torch.bmm(one_hot.transpose(1, 2), coords_sq) / safe_counts # (B, K, 2) -> E[y^2], E[x^2]
        mean_xy = torch.bmm(one_hot.transpose(1, 2), coords_xy) / safe_counts # (B, K, 1) -> E[yx]
        
        # Var(y) = E[y^2] - E[y]^2
        var_y = mean_sq[:, :, 0:1] - centroids[:, :, 0:1] ** 2
        var_x = mean_sq[:, :, 1:2] - centroids[:, :, 1:2] ** 2
        cov_yx = mean_xy - centroids[:, :, 0:1] * centroids[:, :, 1:2]
        
        shape_desc = torch.cat([var_y, var_x, cov_yx], dim=2) # (B, K, 3)
        
        # --- Compute Color Statistics ---
        if images is not None:
            # Resize images to feature map size for consistent pooling
            # Note: 'images' here are typically normalized (mean~0, std~1) if coming from DETR pipeline.
            # The MLP can learn to handle this, but be aware that color stats will be in normalized space.
            images_resized = F.interpolate(images, size=(H, W), mode='bilinear', align_corners=False)
            img_flat = images_resized.flatten(2).transpose(1, 2) # (B, H*W, 3)
            
            # Mean Color
            color_mean = torch.bmm(one_hot.transpose(1, 2), img_flat) / safe_counts # (B, K, 3)
            
            # Std Color
            img_sq_flat = img_flat ** 2
            color_sq_mean = torch.bmm(one_hot.transpose(1, 2), img_sq_flat) / safe_counts
            color_var = color_sq_mean - color_mean ** 2
            color_std = torch.sqrt(torch.clamp(color_var, min=1e-6)) # (B, K, 3)
        else:
            # Fallback if images not provided
            color_mean = torch.zeros(B, K, 3, device=x.device)
            color_std = torch.zeros(B, K, 3, device=x.device)
        
        # --- Compute Fourier Shape Descriptors ---
        if self.use_fourier_shape and self.fourier_shape is not None:
            fourier_desc = self.fourier_shape(superpixel_map, centroids, H, W)
        else:
            fourier_desc = None
            
        # --- Fuse Features ---
        # [CNN(C), ColorMean(3), ColorStd(3), CovarianceShape(3), Centroid(2), FourierShape(n_coeffs)]
        extra_features = torch.cat([color_mean, color_std, shape_desc, centroids], dim=2)
        if fourier_desc is not None:
            extra_features = torch.cat([extra_features, fourier_desc], dim=2)
        combined_features = torch.cat([pooled_features, extra_features], dim=2)
        
        pooled_features = self.fusion_mlp(combined_features)
        
        return pooled_features, mask, centroids
