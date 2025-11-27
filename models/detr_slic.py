import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

from util.misc import NestedTensor, nested_tensor_from_tensor_list
from .backbone import build_backbone
from .transformer import build_transformer, TransformerDecoder
from .superpixel.pool import SuperpixelPool
from .superpixel.sca import SCA
from .superpixel.superpixel_encoder import SuperpixelTransformerEncoder
from .detr import MLP, SetCriterion, PostProcess
from .matcher import build_matcher


class SinusoidalPositionEmbedding2D(nn.Module):
    """
    Sinusoidal positional encoding for 2D coordinates (like superpixel centroids).
    Same approach as DETR's PositionEmbeddingSine but for arbitrary (y, x) coordinates.
    """
    def __init__(self, num_pos_feats=128, temperature=10000, scale=2 * math.pi):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.scale = scale
        
    def forward(self, centroids):
        """
        Args:
            centroids: (B, K, 2) with normalized (y, x) coordinates in [0, 1]
        Returns:
            pos: (B, K, num_pos_feats * 2) positional embeddings
        """
        # centroids[:, :, 0] = y, centroids[:, :, 1] = x
        # Scale to [0, 2*pi]
        y_embed = centroids[:, :, 0:1] * self.scale  # (B, K, 1)
        x_embed = centroids[:, :, 1:2] * self.scale  # (B, K, 1)
        
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=centroids.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        
        # (B, K, num_pos_feats)
        pos_x = x_embed / dim_t
        pos_y = y_embed / dim_t
        
        # Apply sin/cos
        pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
        pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
        
        # Concatenate y and x embeddings
        pos = torch.cat((pos_y, pos_x), dim=2)  # (B, K, num_pos_feats * 2)
        
        return pos


class DETRSlic(nn.Module):
    """
    DETR with SLIC Superpixel-based encoder for efficient object detection.
    """
    def __init__(self,
                 backbone,
                 transformer,
                 num_classes: int,
                 num_queries: int,
                 n_superpixels: int = 100,
                 slic_compactness: float = 10.0,
                 pooling_method: str = 'mean',
                 use_superpixel_encoder: bool = True,
                 aux_loss: bool = False,
                 use_sca: bool = False,
                 use_fourier_shape: bool = True,
                 n_fourier_coeffs: int = 16):
        super().__init__()
        self.num_queries = num_queries
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.use_superpixel_encoder = use_superpixel_encoder
        self.use_sca = use_sca
        
        hidden_dim = transformer.d_model
        
        # Feature pooling module
        self.superpixel_pool = SuperpixelPool(
            in_dim=hidden_dim,
            out_dim=hidden_dim,
            max_superpixels=n_superpixels,
            pooling_type=pooling_method,
            use_fourier_shape=use_fourier_shape,
            n_fourier_coeffs=n_fourier_coeffs
        )
        if use_sca:
            self.sca = SCA(dim=hidden_dim)
        
        # Sinusoidal Positional Encoding for superpixel centroids
        # Uses the same proven approach as DETR's position encoding
        self.pos_embed = SinusoidalPositionEmbedding2D(num_pos_feats=hidden_dim // 2)
        
        # Projection layer for backbone features
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        
        if use_superpixel_encoder:
            # Use superpixel-based encoder
            self.superpixel_encoder = SuperpixelTransformerEncoder(
                d_model=hidden_dim,
                nhead=transformer.nhead,
                num_encoder_layers=transformer.encoder.num_layers,
                dim_feedforward=2048,
                dropout=0.1,
                normalize_before=False
            )
            # Keep original decoder
            self.decoder = transformer.decoder
        else:
            # Use full transformer
            self.transformer = transformer
        
        # Detection heads
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        
        # Use default PyTorch initialization (same as official DETR)
        # No custom bias initialization - let the model learn from scratch
        
    def forward(self, samples: NestedTensor, targets: Optional[list] = None):
        """
        Args:
            samples: NestedTensor with images and masks
            targets: Optional list of targets (containing 'slic_map' if pre-computed)
        Returns:
            Dictionary with predictions
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        
        # Extract features from backbone
        features, pos = self.backbone(samples)
        src, mask = features[-1].decompose()
        
        # Project features
        src_proj = self.input_proj(src)  # (B, hidden_dim, H, W)
        
        B, C, H_feat, W_feat = src_proj.shape
        
        # Determine superpixel maps
        if targets is not None and 'slic_map' in targets[0]:
            # Use pre-computed superpixels from targets
            slic_maps_list = [t['slic_map'] for t in targets]
            
            # Pad maps to match batch dimensions
            # We use a custom padding value of -1 to indicate padding
            max_h, max_w = samples.tensors.shape[-2:]
            slic_maps_padded = torch.full((B, max_h, max_w), -1, dtype=torch.long, device=samples.tensors.device)
            
            num_superpixels_list = []
            for i, sp_map in enumerate(slic_maps_list):
                h, w = sp_map.shape
                slic_maps_padded[i, :h, :w] = sp_map
                num_superpixels_list.append(sp_map.max() + 1)
            
            superpixel_maps = slic_maps_padded
            num_superpixels = torch.tensor(num_superpixels_list, device=samples.tensors.device)
            
            # Resize superpixel maps to feature map size using nearest neighbor
            superpixel_maps = superpixel_maps.float().unsqueeze(1)  # (B, 1, H, W)
            superpixel_maps = F.interpolate(superpixel_maps, size=(H_feat, W_feat), mode='nearest')
            superpixel_maps = superpixel_maps.squeeze(1).long()  # (B, H_feat, W_feat)
            
        else:
            # Pre-computed superpixels are required
            raise RuntimeError(
                "Pre-computed superpixel maps ('slic_map') not found in targets. "
                "Please run 'util/generate_superpixel.py' to generate them offline before training. "
                "On-the-fly generation is disabled to ensure quality."
            )
        
        # Pool features within superpixels
        # Pass original images for color stats calculation
        superpixel_features, superpixel_mask, centroids = self.superpixel_pool(
            src_proj, superpixel_maps, images=samples.tensors
        )
        
        # Sinusoidal positional encoding from centroids
        # (B, K, 2) -> (B, K, hidden_dim)
        superpixel_pos = self.pos_embed(centroids)
        
        if self.use_sca:
            # Apply SCA
            # pixel_feats needs to be (B, H*W, C)
            pixel_feats_flat = src_proj.flatten(2).transpose(1, 2)
            # mask for SCA: True indicates padding. superpixel_mask: True indicates valid.
            # So we pass ~superpixel_mask
            superpixel_features, _ = self.sca(pixel_feats_flat, superpixel_features, mask=~superpixel_mask)
        
        # Prepare for transformer: (N_superpixels, B, C)
        superpixel_features = superpixel_features.permute(1, 0, 2)
        # Use sinusoidal positional encoding
        superpixel_pos = superpixel_pos.permute(1, 0, 2)
        
        # Invert mask for transformer (True = padding, False = valid)
        superpixel_padding_mask = ~superpixel_mask  # (B, max_segments)
        
        if self.use_superpixel_encoder:
            # Use superpixel encoder
            memory = self.superpixel_encoder(
                superpixel_features,
                src_key_padding_mask=superpixel_padding_mask,
                pos=superpixel_pos
            )  # (N_superpixels, B, hidden_dim)
            
            # Decoder
            query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, B, 1)  # (num_queries, B, hidden_dim)
            tgt = torch.zeros_like(query_embed)
            
            hs = self.decoder(
                tgt, memory,
                memory_key_padding_mask=superpixel_padding_mask,
                pos=superpixel_pos,
                query_pos=query_embed
            )  # (num_decoder_layers, num_queries, B, hidden_dim)
            
            hs = hs.transpose(1, 2)  # (num_decoder_layers, B, num_queries, hidden_dim)
        else:
            # Use original transformer (fallback)
            hs = self.transformer(src_proj, mask, self.query_embed.weight, pos[-1])[0]
        
        # Prediction heads
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
    Build DETR-SLIC model with criterion and postprocessors.
    """
    # Use args.num_classes if provided, otherwise default logic
    if hasattr(args, 'num_classes'):
        num_classes = args.num_classes
    else:
        num_classes = 20 if args.dataset_file != 'coco' else 91
        
    device = torch.device(args.device)
    
    backbone = build_backbone(args)
    transformer = build_transformer(args)
    
    model = DETRSlic(
        backbone=backbone,
        transformer=transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        n_superpixels=getattr(args, 'n_superpixels', 100),
        slic_compactness=getattr(args, 'slic_compactness', 10.0),
        pooling_method=getattr(args, 'pooling_method', 'mean'),
        use_superpixel_encoder=getattr(args, 'use_superpixel_encoder', True),
        aux_loss=args.aux_loss,
        use_sca=getattr(args, 'use_sca', False),
        use_fourier_shape=getattr(args, 'use_fourier_shape', True),
        n_fourier_coeffs=getattr(args, 'n_fourier_coeffs', 16)
    )
    
    # Criterion and postprocessors (reuse from DETR)
    matcher = build_matcher(args)
    weight_dict = {'loss_ce': 1, 'loss_bbox': args.bbox_loss_coef, 'loss_giou': args.giou_loss_coef}
    
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)
    
    losses = ['labels', 'boxes', 'cardinality']
    
    # Get focal loss parameters if available
    use_focal_loss = getattr(args, 'use_focal_loss', False)
    focal_alpha = getattr(args, 'focal_alpha', 0.25)
    focal_gamma = getattr(args, 'focal_gamma', 2.0)
    
    criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
                             eos_coef=args.eos_coef, losses=losses,
                             use_focal_loss=use_focal_loss,
                             focal_alpha=focal_alpha,
                             focal_gamma=focal_gamma)
    criterion.to(device)
    
    postprocessors = {'bbox': PostProcess()}
    
    return model, criterion, postprocessors