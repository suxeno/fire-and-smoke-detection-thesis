import torch
import torch.nn as nn

class SCA(nn.Module):
    def __init__(self, dim=256, num_heads=4):
        super().__init__()
        self.p2s_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.s2p_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # Initialize parameters with xavier uniform (same as DETR)
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Initialize all parameters with xavier uniform (same as DETR)."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, pixel_feats, superpixel_feats, mask=None):
        """
        Args:
            pixel_feats: Tensor of shape (B, H*W, C)
            superpixel_feats: Tensor of shape (B, K, C)
            mask: Optional mask for superpixels (B, K). True indicates padding.
        
        Returns:
            new_superpixel_feats: Updated superpixel features
            new_pixel_feats: Updated pixel features
        """
        # P2S Attention: Update superpixels by attending to pixels
        # Query: superpixel_feats, Key: pixel_feats, Value: pixel_feats
        # Output shape: (B, K, C)
        sp_out, _ = self.p2s_attn(superpixel_feats, pixel_feats, pixel_feats)
        new_superpixel_feats = self.norm1(superpixel_feats + sp_out)

        # S2P Attention: Update pixels by attending to superpixels
        # Query: pixel_feats, Key: new_superpixel_feats, Value: new_superpixel_feats
        # Output shape: (B, H*W, C)
        # We use the mask here because we are attending TO superpixels
        p_out, _ = self.s2p_attn(pixel_feats, new_superpixel_feats, new_superpixel_feats, key_padding_mask=mask)
        new_pixel_feats = self.norm2(pixel_feats + p_out)

        return new_superpixel_feats, new_pixel_feats
