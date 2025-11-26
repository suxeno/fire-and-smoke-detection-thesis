import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from util.misc import NestedTensor, nested_tensor_from_tensor_list
from .backbone import build_backbone
from .transformer import build_transformer, TransformerDecoder
from .superpixel.feature_pooling import SuperpixelFeaturePooling
from .superpixel.superpixel_encoder import SuperpixelTransformerEncoder
from .detr import MLP, SetCriterion, PostProcess
from .matcher import build_matcher


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
                 aux_loss: bool = False):
        super().__init__()
        self.num_queries = num_queries
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.use_superpixel_encoder = use_superpixel_encoder
        
        hidden_dim = transformer.d_model
        
        # Feature pooling module
        self.superpixel_pooling = SuperpixelFeaturePooling(
            pooling_method=pooling_method,
            add_spatial_encoding=True,
            feature_dim=hidden_dim
        )
        
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
        
        # Positional encoding for superpixels
        self.superpixel_pos_embed = nn.Linear(4, hidden_dim)  # spatial features -> pos encoding
        
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
        superpixel_features, superpixel_mask = self.superpixel_pooling(
            src_proj, superpixel_maps, num_superpixels
        )  # (B, max_segments, hidden_dim), (B, max_segments)
        
        # Compute positional encoding for superpixels
        spatial_features = self.superpixel_pooling.compute_spatial_features(
            superpixel_maps, num_superpixels
        )
        superpixel_pos = self.superpixel_pos_embed(spatial_features)  # (B, max_segments, hidden_dim)
        
        # Prepare for transformer: (N_superpixels, B, C)
        superpixel_features = superpixel_features.permute(1, 0, 2)
        superpixel_pos = superpixel_pos.permute(1, 0, 2)
        
        # Invert mask for transformer (True = valid, False = padding)
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
        aux_loss=args.aux_loss
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
    criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
                             eos_coef=args.eos_coef, losses=losses)
    criterion.to(device)
    
    postprocessors = {'bbox': PostProcess()}
    
    return model, criterion, postprocessors