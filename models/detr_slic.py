"""
DETR-SLIC: Superpixel-based Object Detection WITHOUT CNN Backbone

This module implements a lightweight detection model that uses superpixel
tokenization instead of CNN features. Ideal for fire/smoke detection where
color and texture are highly discriminative.

Pipeline:
1. SLIC superpixel segmentation (pre-computed offline)
2. Extract 19D features per superpixel (color, gradient, shape, position)
3. Project to hidden_dim using learnable MLP
4. Transformer encoder for superpixel interaction  
5. Transformer decoder with object queries
6. Detection heads (class + bbox)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from util.misc import NestedTensor, nested_tensor_from_tensor_list
from .transformer import TransformerDecoder, TransformerDecoderLayer
from .position_encoding import PositionEmbeddingCoordinate
from .superpixel.pool import SuperpixelPool
from .superpixel.superpixel_encoder import SuperpixelTransformerEncoder
from .detr import MLP, SetCriterion, PostProcess
from .matcher import build_matcher


class DETRSlic(nn.Module):
    """
    DETR with SLIC Superpixel tokenization - NO CNN BACKBONE.
    
    Extracts features directly from image pixels using superpixel pooling,
    then processes with transformer encoder-decoder for detection.
    
    Args:
        num_classes: Number of object classes (excluding background)
        num_queries: Number of object queries
        hidden_dim: Transformer model dimension
        n_superpixels: Maximum number of superpixels per image
        nheads: Number of attention heads
        num_encoder_layers: Number of encoder layers
        num_decoder_layers: Number of decoder layers
        dim_feedforward: FFN dimension
        dropout: Dropout rate
        aux_loss: Use auxiliary decoding losses
    """
    
    def __init__(self,
                 num_classes: int,
                 num_queries: int = 100,
                 hidden_dim: int = 256,
                 n_superpixels: int = 300,
                 nheads: int = 8,
                 num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 aux_loss: bool = False,
                 use_multiscale: bool = False,
                 superpixel_scales: list = None):
        super().__init__()
        
        self.num_queries = num_queries
        self.aux_loss = aux_loss
        self.hidden_dim = hidden_dim
        self.use_multiscale = use_multiscale
        self.superpixel_scales = superpixel_scales or [150, 300, 600]
        
        # Superpixel feature extraction (69D -> hidden_dim, or 207D for multi-scale)
        # No CNN backbone - extracts features directly from image pixels
        self.superpixel_pool = SuperpixelPool(
            hidden_dim=hidden_dim,
            max_superpixels=n_superpixels,
            use_cnn_features=False,
            use_multiscale=use_multiscale,
            superpixel_scales=superpixel_scales
        )
        
        # Positional encoding for superpixel centroids
        self.pos_embed = PositionEmbeddingCoordinate(num_pos_feats=hidden_dim // 2)
        
        # Transformer encoder for superpixel tokens
        self.superpixel_encoder = SuperpixelTransformerEncoder(
            d_model=hidden_dim,
            nhead=nheads,
            num_encoder_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            normalize_before=False
        )
        
        # Transformer decoder
        decoder_layer = TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=nheads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            normalize_before=False
        )
        decoder_norm = nn.LayerNorm(hidden_dim)
        self.decoder = TransformerDecoder(
            decoder_layer, 
            num_decoder_layers, 
            decoder_norm,
            return_intermediate=True
        )
        
        # Detection heads
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
    
    def _resize_slic_map(self, sp_map: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
        """
        Resize superpixel map to match target image size using nearest-neighbor.
        
        Args:
            sp_map: (H_orig, W_orig) superpixel map (long tensor)
            target_h: Target height
            target_w: Target width
            
        Returns:
            Resized superpixel map (target_h, target_w)
        """
        h, w = sp_map.shape
        if h == target_h and w == target_w:
            return sp_map
        
        # Use nearest-neighbor interpolation to preserve superpixel IDs
        sp_map_float = sp_map.float().unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        resized = F.interpolate(sp_map_float, size=(target_h, target_w), mode='nearest')
        return resized.squeeze(0).squeeze(0).long()
        
    def forward(self, samples: NestedTensor, targets: Optional[list] = None):
        """
        Args:
            samples: NestedTensor with images and masks
            targets: List of targets with 'slic_map' key (pre-computed superpixels)
        Returns:
            Dictionary with 'pred_logits' and 'pred_boxes'
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        
        images = samples.tensors  # (B, 3, H, W)
        B, _, H, W = images.shape
        
        # Get pre-computed superpixel maps
        if targets is None or 'slic_map' not in targets[0]:
            raise RuntimeError(
                "Pre-computed superpixel maps required in targets['slic_map']. "
                "Run 'util/generate_superpixel.py' to generate them before training."
            )
        
        # Pad superpixel maps to match image size (handles variable sizes)
        # Note: superpixel maps may need resizing if image was resized by augmentation
        slic_maps = torch.full((B, H, W), -1, dtype=torch.long, device=images.device)
        for i, t in enumerate(targets):
            sp_map = t['slic_map'].to(images.device)
            # Resize if dimensions don't match (due to data augmentation)
            sp_map = self._resize_slic_map(sp_map, H, W)
            slic_maps[i] = sp_map
        
        # Prepare multi-scale maps if using multi-scale mode
        multiscale_maps = None
        if self.use_multiscale:
            multiscale_maps = {}
            for scale in self.superpixel_scales:
                key = f'slic_{scale}'
                if key in targets[0]:
                    scale_maps = torch.full((B, H, W), -1, dtype=torch.long, device=images.device)
                    for i, t in enumerate(targets):
                        sp_map = t[key].to(images.device)
                        # Resize if dimensions don't match
                        sp_map = self._resize_slic_map(sp_map, H, W)
                        scale_maps[i] = sp_map
                    multiscale_maps[key] = scale_maps
        
        # Extract 69D features from superpixels (no CNN!)
        superpixel_features, mask, centroids = self.superpixel_pool(
            images, slic_maps, multiscale_maps=multiscale_maps
        )
        # superpixel_features: (B, K, hidden_dim)
        # mask: (B, K) - True for valid superpixels
        # centroids: (B, K, 2) - normalized coordinates
        
        # Positional encoding from centroids
        superpixel_pos = self.pos_embed(centroids)  # (B, K, hidden_dim)
        
        # Prepare for transformer: (K, B, hidden_dim)
        superpixel_features = superpixel_features.permute(1, 0, 2)
        superpixel_pos = superpixel_pos.permute(1, 0, 2)
        
        # Padding mask (True = padding)
        superpixel_padding_mask = ~mask
        
        # Encoder
        memory = self.superpixel_encoder(
            superpixel_features,
            src_key_padding_mask=superpixel_padding_mask,
            pos=superpixel_pos
        )  # (K, B, hidden_dim)
        
        # Decoder
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, B, 1)
        tgt = torch.zeros_like(query_embed)
        
        hs = self.decoder(
            tgt, memory,
            memory_key_padding_mask=superpixel_padding_mask,
            pos=superpixel_pos,
            query_pos=query_embed
        )  # (num_layers, num_queries, B, hidden_dim)
        
        hs = hs.transpose(1, 2)  # (num_layers, B, num_queries, hidden_dim)
        
        # Detection heads
        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        
        out = {
            'pred_logits': outputs_class[-1],
            'pred_boxes': outputs_coord[-1]
        }
        
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        
        return out
    
    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


def build_detr_slic(args):
    """
    Build DETR-SLIC model.
    """
    if hasattr(args, 'num_classes'):
        num_classes = args.num_classes
    else:
        num_classes = 2  # fire, smoke
        
    device = torch.device(args.device)
    
    model = DETRSlic(
        num_classes=num_classes,
        num_queries=args.num_queries,
        hidden_dim=getattr(args, 'hidden_dim', 256),
        n_superpixels=getattr(args, 'n_superpixels', 300),
        nheads=getattr(args, 'nheads', 8),
        num_encoder_layers=getattr(args, 'enc_layers', 6),
        num_decoder_layers=getattr(args, 'dec_layers', 6),
        dim_feedforward=getattr(args, 'dim_feedforward', 2048),
        dropout=getattr(args, 'dropout', 0.1),
        aux_loss=args.aux_loss,
        use_multiscale=getattr(args, 'use_multiscale', False),
        superpixel_scales=getattr(args, 'superpixel_scales', [150, 300, 600])
    )
    
    # Criterion and postprocessors
    matcher = build_matcher(args)
    weight_dict = {'loss_ce': 1, 'loss_bbox': args.bbox_loss_coef, 'loss_giou': args.giou_loss_coef}
    
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)
    
    losses = ['labels', 'boxes', 'cardinality']
    
    criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
                             eos_coef=args.eos_coef, losses=losses)
    criterion.to(device)
    
    postprocessors = {'bbox': PostProcess()}
    
    return model, criterion, postprocessors