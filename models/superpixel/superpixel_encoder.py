import torch
import torch.nn as nn
from typing import Optional
from torch import Tensor

from ..transformer import TransformerEncoder, TransformerEncoderLayer


class SuperpixelTransformerEncoder(nn.Module):
    """
    Transformer encoder that operates on superpixel tokens instead of pixel-level features.
    """
    def __init__(self,
                 d_model: int = 256,
                 nhead: int = 8,
                 num_encoder_layers: int = 6,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 activation: str = "relu",
                 normalize_before: bool = False):
        super().__init__()
        
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        
        self.d_model = d_model
        
    def forward(self,
                src: Tensor,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        """
        Args:
            src: (N_superpixels, B, C) superpixel features
            mask: optional attention mask
            src_key_padding_mask: (B, N_superpixels) mask for padded superpixels
            pos: (N_superpixels, B, C) positional encoding
        Returns:
            memory: (N_superpixels, B, C) encoded features
        """
        return self.encoder(src, mask=mask, src_key_padding_mask=src_key_padding_mask, pos=pos)