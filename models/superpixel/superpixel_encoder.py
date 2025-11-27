import torch
import torch.nn as nn
from typing import Optional
from torch import Tensor

from ..transformer import TransformerEncoder, TransformerEncoderLayer


class SuperpixelTransformerEncoder(nn.Module):
    """
    Transformer encoder that operates on superpixel tokens instead of pixel-level features.
    
    Includes a residual connection from input to output to preserve token diversity,
    which is critical for the decoder to produce diverse predictions.
    """
    def __init__(self,
                 d_model: int = 256,
                 nhead: int = 8,
                 num_encoder_layers: int = 6,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 activation: str = "relu",
                 normalize_before: bool = False,
                 use_residual: bool = True):
        super().__init__()
        
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        
        self.d_model = d_model
        self.use_residual = use_residual
        
        # Learnable mixing weight for residual connection
        # Allows model to learn how much of the original features to preserve
        # Initialize to prefer original features (sigmoid(-1) ≈ 0.27 for encoded, 0.73 for input)
        if use_residual:
            self.residual_weight = nn.Parameter(torch.tensor(-1.0))
        
        # Final layer norm after residual
        self.output_norm = nn.LayerNorm(d_model)
        
        # Initialize parameters like DETR's transformer
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Initialize all parameters with xavier uniform (same as DETR)."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
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
            memory: (N_superpixels, B, C) encoded features with preserved diversity
        """
        # Standard encoder forward
        encoded = self.encoder(src, mask=mask, src_key_padding_mask=src_key_padding_mask, pos=pos)
        
        if self.use_residual:
            # Residual connection from input to preserve token diversity
            # This prevents the encoder from collapsing all tokens to the same representation
            alpha = torch.sigmoid(self.residual_weight)  # Keep in [0, 1]
            output = alpha * encoded + (1 - alpha) * src
            output = self.output_norm(output)
        else:
            output = encoded
            
        return output