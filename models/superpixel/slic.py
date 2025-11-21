import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import numpy as np

try:
    from skimage.segmentation import slic
    SLIC_AVAILABLE = True
except ImportError:
    SLIC_AVAILABLE = False


class SLICSuperpixel(nn.Module):
    """
    SLIC Superpixel generation module.
    Can be differentiable using soft assignment or non-differentiable using skimage SLIC.
    """
    def __init__(self, 
                 n_segments: int = 100,
                 compactness: float = 10.0,
                 sigma: float = 1.0,
                 differentiable: bool = False):
        super().__init__()
        self.n_segments = n_segments
        self.compactness = compactness
        self.sigma = sigma
        self.differentiable = differentiable
        
    @torch.no_grad()
    def forward_slic(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Non-differentiable SLIC using skimage.
        Args:
            images: (B, 3, H, W) normalized images
        Returns:
            superpixel_maps: (B, H, W) with superpixel IDs
            num_superpixels: (B,) actual number of superpixels per image
        """
        if not SLIC_AVAILABLE:
            raise RuntimeError("skimage not available. Install with: pip install scikit-image")
        
        B, C, H, W = images.shape
        device = images.device
        
        # Convert to numpy for SLIC
        images_np = images.cpu().permute(0, 2, 3, 1).numpy()
        
        superpixel_maps = []
        num_superpixels_list = []
        
        for i in range(B):
            # Denormalize image for SLIC (expects [0, 1] range)
            img = images_np[i]
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            
            # Run SLIC
            segments = slic(img, 
                          n_segments=self.n_segments,
                          compactness=self.compactness,
                          sigma=self.sigma,
                          start_label=0)
            
            superpixel_maps.append(torch.from_numpy(segments))
            num_superpixels_list.append(segments.max() + 1)
        
        superpixel_maps = torch.stack(superpixel_maps).to(device)
        num_superpixels = torch.tensor(num_superpixels_list, device=device)
        
        return superpixel_maps, num_superpixels
    
    def forward_soft_slic(self, images: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """
        Differentiable soft superpixel assignment using learnable cluster centers.
        This is a simplified version for gradient flow.
        Args:
            images: (B, 3, H, W)
            features: (B, C, H, W) from backbone
        Returns:
            soft_assignment: (B, N_segments, H, W) soft assignment scores
        """
        # TODO: Implement differentiable SLIC or soft clustering
        # For now, use regular SLIC
        return self.forward_slic(images)
    
    def forward(self, images: torch.Tensor, features: torch.Tensor = None):
        if self.differentiable and features is not None:
            return self.forward_soft_slic(images, features)
        else:
            return self.forward_slic(images)