import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class SuperpixelFeaturePooling(nn.Module):
    """
    Pool backbone features within each superpixel to create superpixel-level tokens.
    """
    def __init__(self, 
                 pooling_method: str = 'mean',
                 add_spatial_encoding: bool = True,
                 feature_dim: int = 256):
        super().__init__()
        self.pooling_method = pooling_method
        self.add_spatial_encoding = add_spatial_encoding
        self.feature_dim = feature_dim
        
        if add_spatial_encoding:
            # Learnable projection for spatial features
            self.spatial_proj = nn.Linear(4, feature_dim)  # [center_x, center_y, width, height]
    
    def pool_features(self, 
                     features: torch.Tensor,
                     superpixel_maps: torch.Tensor,
                     num_superpixels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Pool features for each superpixel.
        Args:
            features: (B, C, H, W) backbone features
            superpixel_maps: (B, H, W) superpixel IDs
            num_superpixels: (B,) number of superpixels per image
        Returns:
            pooled_features: (B, max_segments, C) pooled features
            valid_mask: (B, max_segments) mask for valid superpixels
        """
        B, C, H, W = features.shape
        max_segments = num_superpixels.max().item()
        device = features.device
        
        pooled_features = torch.zeros(B, max_segments, C, device=device)
        valid_mask = torch.zeros(B, max_segments, dtype=torch.bool, device=device)
        
        for b in range(B):
            n_seg = num_superpixels[b].item()
            valid_mask[b, :n_seg] = True
            
            for s in range(n_seg):
                # Create mask for current superpixel
                mask = (superpixel_maps[b] == s)  # (H, W)
                
                if mask.sum() == 0:
                    continue
                
                # Expand mask to match feature dimensions
                mask_expanded = mask.unsqueeze(0).expand(C, -1, -1)  # (C, H, W)
                
                # Extract features for this superpixel
                superpixel_feats = features[b].masked_select(mask_expanded).view(C, -1)  # (C, N_pixels)
                
                # Pool features
                if self.pooling_method == 'mean':
                    pooled_features[b, s] = superpixel_feats.mean(dim=1)
                elif self.pooling_method == 'max':
                    pooled_features[b, s] = superpixel_feats.max(dim=1)[0]
                elif self.pooling_method == 'sum':
                    pooled_features[b, s] = superpixel_feats.sum(dim=1)
        
        return pooled_features, valid_mask
    
    def compute_spatial_features(self,
                                 superpixel_maps: torch.Tensor,
                                 num_superpixels: torch.Tensor) -> torch.Tensor:
        """
        Compute spatial statistics for each superpixel.
        Args:
            superpixel_maps: (B, H, W)
            num_superpixels: (B,)
        Returns:
            spatial_features: (B, max_segments, 4) [center_x, center_y, width, height] normalized
        """
        B, H, W = superpixel_maps.shape
        max_segments = num_superpixels.max().item()
        device = superpixel_maps.device
        
        spatial_features = torch.zeros(B, max_segments, 4, device=device)
        
        # Create coordinate grids
        y_coords = torch.arange(H, device=device).view(H, 1).expand(H, W).float()
        x_coords = torch.arange(W, device=device).view(1, W).expand(H, W).float()
        
        for b in range(B):
            n_seg = num_superpixels[b].item()
            
            for s in range(n_seg):
                mask = (superpixel_maps[b] == s)
                
                if mask.sum() == 0:
                    continue
                
                y_pixels = y_coords[mask]
                x_pixels = x_coords[mask]
                
                # Compute center (normalized to [0, 1])
                center_y = y_pixels.mean() / H
                center_x = x_pixels.mean() / W
                
                # Compute bounding box (normalized)
                y_min, y_max = y_pixels.min() / H, y_pixels.max() / H
                x_min, x_max = x_pixels.min() / W, x_pixels.max() / W
                
                height = y_max - y_min
                width = x_max - x_min
                
                spatial_features[b, s] = torch.tensor([center_x, center_y, width, height], 
                                                      device=device)
        
        return spatial_features
    
    def forward(self,
                features: torch.Tensor,
                superpixel_maps: torch.Tensor,
                num_superpixels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features: (B, C, H, W)
            superpixel_maps: (B, H, W)
            num_superpixels: (B,)
        Returns:
            superpixel_features: (B, max_segments, C) or (B, max_segments, C+feature_dim)
            valid_mask: (B, max_segments)
        """
        # Pool features
        pooled_features, valid_mask = self.pool_features(features, superpixel_maps, num_superpixels)
        
        # Add spatial encoding if enabled
        if self.add_spatial_encoding:
            spatial_features = self.compute_spatial_features(superpixel_maps, num_superpixels)
            spatial_encoding = self.spatial_proj(spatial_features)
            pooled_features = pooled_features + spatial_encoding
        
        return pooled_features, valid_mask