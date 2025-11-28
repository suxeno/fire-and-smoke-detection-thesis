"""
DETR-Hybrid: CNN Backbone + Superpixel Features

This module implements a hybrid detection model combining:
1. CNN backbone (ResNet) for rich semantic features
2. SLIC superpixels for efficient tokenization and handcrafted features

Pipeline:
1. CNN backbone extracts feature map
2. SLIC superpixels pool CNN features per region
3. Extract 19D handcrafted features (color, gradient, shape, position)
4. Fuse CNN features with handcrafted features
5. Transformer encoder-decoder for detection
6. Detection heads (class + bbox)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from util.misc import NestedTensor, nested_tensor_from_tensor_list
from .backbone import build_backbone
from .transformer import build_transformer, TransformerDecoder, TransformerDecoderLayer
from .position_encoding import PositionEmbeddingCoordinate
from .superpixel.pool import SuperpixelPool
from .superpixel.superpixel_encoder import SuperpixelTransformerEncoder
from .detr import MLP, SetCriterion, PostProcess
from .matcher import build_matcher


class DETRHybrid(nn.Module):
    """
    DETR with CNN backbone + SLIC Superpixel features.
    
    Combines learned CNN features with handcrafted superpixel descriptors
    for robust object detection.
    
    Args:
        backbone: CNN backbone (ResNet)
        transformer: Transformer encoder-decoder (only decoder used)
        num_classes: Number of object classes
        num_queries: Number of object queries
        n_superpixels: Maximum superpixels per image
        aux_loss: Use auxiliary decoding losses
        use_multiscale: Use multi-scale superpixels
        superpixel_scales: List of superpixel scales
    """
    
    def __init__(self,
                 backbone,
                 transformer,
                 num_classes: int,
                 num_queries: int,
                 n_superpixels: int = 300,
                 aux_loss: bool = False,
                 use_multiscale: bool = False,
                 superpixel_scales: list = None):
        super().__init__()
        
        self.num_queries = num_queries
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.use_multiscale = use_multiscale
        self.superpixel_scales = superpixel_scales or [150, 300, 600]
        
        hidden_dim = transformer.d_model
        
        # Projection layer for backbone features
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        
        # Superpixel pooling with CNN feature fusion
        # Pools CNN features + extracts 69D handcrafted features, then fuses
        self.superpixel_pool = SuperpixelPool(
            hidden_dim=hidden_dim,
            max_superpixels=n_superpixels,
            use_cnn_features=True,  # Fuse CNN with 69D handcrafted features
            cnn_dim=hidden_dim,
            use_multiscale=use_multiscale,
            superpixel_scales=superpixel_scales
        )
        
        # Positional encoding for superpixel centroids
        self.pos_embed = PositionEmbeddingCoordinate(num_pos_feats=hidden_dim // 2)
        
        # Superpixel transformer encoder
        self.superpixel_encoder = SuperpixelTransformerEncoder(
            d_model=hidden_dim,
            nhead=transformer.nhead,
            num_encoder_layers=transformer.encoder.num_layers,
            dim_feedforward=2048,
            dropout=0.1,
            normalize_before=False
        )
        
        # Use transformer's decoder
        self.decoder = transformer.decoder
        
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
            targets: List of targets with 'slic_map' key
        Returns:
            Dictionary with 'pred_logits' and 'pred_boxes'
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        
        # Extract features from CNN backbone
        features, pos = self.backbone(samples)
        src, mask = features[-1].decompose()
        
        # Project to hidden_dim
        src_proj = self.input_proj(src)  # (B, hidden_dim, H_feat, W_feat)
        
        B, C, H_feat, W_feat = src_proj.shape
        
        # Get pre-computed superpixel maps
        if targets is None or 'slic_map' not in targets[0]:
            raise RuntimeError(
                "Pre-computed superpixel maps required in targets['slic_map']. "
                "Run 'util/generate_superpixel.py' to generate them offline."
            )
        
        # Pad superpixel maps to match image size
        # Note: superpixel maps may need resizing if image was resized by augmentation
        max_h, max_w = samples.tensors.shape[-2:]
        slic_maps = torch.full((B, max_h, max_w), -1, dtype=torch.long, device=samples.tensors.device)
        for i, t in enumerate(targets):
            sp_map = t['slic_map'].to(samples.tensors.device)
            # Resize if dimensions don't match (due to data augmentation)
            sp_map = self._resize_slic_map(sp_map, max_h, max_w)
            slic_maps[i] = sp_map
        
        # Prepare multi-scale maps if using multi-scale mode
        multiscale_maps = None
        if self.use_multiscale:
            multiscale_maps = {}
            for scale in self.superpixel_scales:
                key = f'slic_{scale}'
                if key in targets[0]:
                    scale_maps = torch.full((B, max_h, max_w), -1, dtype=torch.long, device=samples.tensors.device)
                    for i, t in enumerate(targets):
                        sp_map = t[key].to(samples.tensors.device)
                        # Resize if dimensions don't match
                        sp_map = self._resize_slic_map(sp_map, max_h, max_w)
                        scale_maps[i] = sp_map
                    multiscale_maps[key] = scale_maps
        
        # Pool CNN features within superpixels and fuse with 69D handcrafted features
        superpixel_features, superpixel_mask, centroids = self.superpixel_pool(
            images=samples.tensors,
            superpixel_map=slic_maps,
            cnn_features=src_proj,
            multiscale_maps=multiscale_maps
        )
        # superpixel_features: (B, K, hidden_dim) - fused CNN + 69D features
        # superpixel_mask: (B, K) - True for valid superpixels
        # centroids: (B, K, 2) - normalized coordinates
        
        # Positional encoding from centroids
        superpixel_pos = self.pos_embed(centroids)  # (B, K, hidden_dim)
        
        # Prepare for transformer: (K, B, hidden_dim)
        superpixel_features = superpixel_features.permute(1, 0, 2)
        superpixel_pos = superpixel_pos.permute(1, 0, 2)
        
        # Padding mask (True = padding)
        superpixel_padding_mask = ~superpixel_mask
        
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


def build_detr_hybrid(args):
    """
    Build DETR-Hybrid model (CNN backbone + superpixel features).
    """
    if hasattr(args, 'num_classes'):
        num_classes = args.num_classes
    else:
        num_classes = 2  # fire, smoke
        
    device = torch.device(args.device)
    
    backbone = build_backbone(args)
    transformer = build_transformer(args)
    
    model = DETRHybrid(
        backbone=backbone,
        transformer=transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        n_superpixels=getattr(args, 'n_superpixels', 300),
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
