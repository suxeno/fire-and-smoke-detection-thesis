# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn

try:
    import cv2  # type: ignore
    _HAS_CV2 = True
except Exception:
    cv2 = None
    _HAS_CV2 = False

from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)

from .backbone import build_backbone
from .matcher import build_matcher
from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
                           dice_loss, sigmoid_focal_loss)
from .transformer import build_transformer


class DETR(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(
        self,
        backbone,
        transformer,
        num_classes,
        num_queries,
        aux_loss=False,
        slic_n_segments=200,
        hybrid_token_mode='mixed',
        pixel_prune: bool = False,
        pixel_prune_keep_ratio: float = 0.8,
        pixel_prune_score_mode: str = 'saliency',
        pixel_prune_w_feature: float = 0.45,
        pixel_prune_w_color: float = 0.25,
        pixel_prune_w_texture: float = 0.20,
        pixel_prune_w_size: float = 0.10,
    ):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.slic_n_segments = slic_n_segments
        self.hybrid_token_mode = hybrid_token_mode
        self.pixel_prune = pixel_prune
        self.pixel_prune_keep_ratio = float(pixel_prune_keep_ratio)
        self.pixel_prune_keep_ratio = max(0.6, min(0.8, self.pixel_prune_keep_ratio))
        self.pixel_prune_score_mode = pixel_prune_score_mode
        self.pixel_prune_w_feature = pixel_prune_w_feature
        self.pixel_prune_w_color = pixel_prune_w_color
        self.pixel_prune_w_texture = pixel_prune_w_texture
        self.pixel_prune_w_size = pixel_prune_w_size
        self.register_buffer(
            'input_mean',
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            'input_std',
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

    def _downsample_slic_map(self, s_map, valid_H: int, valid_W: int, device: torch.device) -> torch.Tensor:
        """Downsample a single superpixel map to feature map resolution.
        
        Uses OpenCV (CPU) nearest-neighbor when available to avoid
        transferring the full-resolution map to GPU. Only the small
        downsampled result is moved to the target device.
        """
        if _HAS_CV2 and torch.is_tensor(s_map) and s_map.device.type == 'cpu':
            sp_np = s_map.numpy().astype('int32', copy=False)
            resized = cv2.resize(sp_np, (valid_W, valid_H), interpolation=cv2.INTER_NEAREST)
            return torch.from_numpy(resized).to(device=device, dtype=torch.long, non_blocking=True)
        else:
            s_map_down = F.interpolate(
                s_map[None, None].float(),
                size=(valid_H, valid_W),
                mode='nearest'
            )[0, 0].long()
            if s_map_down.device != device:
                s_map_down = s_map_down.to(device=device, non_blocking=True)
            return s_map_down

    def _build_pixel_superpixel_map(
        self,
        mask: torch.Tensor,
        slic_maps: list,
        n_segments: int,
        H: int,
        W: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Build a [B, H, W] map assigning each pixel to a superpixel ID.
        
        Invalid pixels (padded or out-of-range SP IDs) get -1.
        This is the ONLY place slic_maps are processed — kept minimal.
        """
        B = mask.shape[0]
        sp_map = torch.full((B, H, W), -1, device=device, dtype=torch.long)

        for b in range(B):
            # Valid spatial region from backbone mask
            valid_y = (~mask[b]).any(dim=1)  # [H]
            valid_x = (~mask[b]).any(dim=0)  # [W]
            valid_H = valid_y.sum().item()
            valid_W = valid_x.sum().item()

            if valid_H == 0 or valid_W == 0:
                continue

            s_map = None
            if slic_maps and b < len(slic_maps):
                s_map = slic_maps[b].get(n_segments, None)

            if s_map is None:
                sp_map[b, :valid_H, :valid_W] = 0
            else:
                s_map_down = self._downsample_slic_map(s_map, valid_H, valid_W, device)
                # Clamp invalid IDs
                invalid = (s_map_down < 0) | (s_map_down >= n_segments)
                s_map_down[invalid] = -1
                sp_map[b, :valid_H, :valid_W] = s_map_down

        return sp_map

    def _compute_per_pixel_scores(
        self,
        proj_src: torch.Tensor,
        samples_tensors: torch.Tensor,
        sp_map_flat: torch.Tensor,
        pixel_valid_mask: torch.Tensor,
        n_segments: int,
        score_mode: str,
        debug: bool = False,
    ) -> torch.Tensor:
        """Compute per-pixel importance scores for pruning.
        
        Uses superpixel assignments to aggregate and broadcast scores.
        All operations are fully batched on GPU — no Python loops.
        
        Args:
            proj_src: [B, C, H, W] projected features
            samples_tensors: [B, 3, Himg, Wimg] original input images
            sp_map_flat: [B, H*W] superpixel assignments (-1 = invalid)
            pixel_valid_mask: [B, H*W] True = valid pixel
            n_segments: number of superpixel segments
            score_mode: 'feature_norm', 'saliency', or 'counts'
            
        Returns:
            scores: [B, H*W] per-pixel scores (invalid pixels get -inf)
        """
        B, C, H, W = proj_src.shape
        device = proj_src.device
        feat_flat = proj_src.flatten(2)  # [B, C, H*W]
        
        # Per-pixel feature norm — always useful and cheap
        pixel_feat_norm = feat_flat.norm(dim=1)  # [B, H*W]

        if score_mode == 'feature_norm':
            scores = pixel_feat_norm
            scores = scores.masked_fill(~pixel_valid_mask, float('-inf'))
            return scores

        # For 'saliency' and 'counts', we need superpixel-level aggregation
        # Compute superpixel counts (fully vectorized)
        valid_sp_mask = pixel_valid_mask & (sp_map_flat >= 0) & (sp_map_flat < n_segments)
        safe_sp_ids = sp_map_flat.clamp(min=0, max=n_segments - 1)  # [B, H*W]

        # Superpixel counts: [B, n_segments]
        sp_counts = torch.zeros((B, n_segments), device=device, dtype=torch.float32)
        sp_counts.scatter_add_(1, safe_sp_ids, valid_sp_mask.float())

        if score_mode == 'counts':
            # Broadcast SP counts back to pixels
            pixel_scores = sp_counts.gather(1, safe_sp_ids)
            pixel_scores = pixel_scores.masked_fill(~pixel_valid_mask, float('-inf'))
            return pixel_scores

        # --- Saliency scoring ---
        # Aggregate per-superpixel feature norms
        sp_feat_sum = torch.zeros((B, n_segments), device=device, dtype=pixel_feat_norm.dtype)
        sp_feat_sum.scatter_add_(1, safe_sp_ids, pixel_feat_norm * valid_sp_mask.float())
        sp_feat_mean = sp_feat_sum / sp_counts.clamp_min(1.0).to(sp_feat_sum.dtype)

        # Color saliency: fire/smoke cues from the input image
        Hf, Wf = H, W
        rgb = samples_tensors
        if rgb.shape[-2:] != (Hf, Wf):
            rgb = F.interpolate(rgb, size=(Hf, Wf), mode='bilinear', align_corners=False)
        rgb = (rgb * self.input_std.to(rgb.dtype)) + self.input_mean.to(rgb.dtype)
        rgb = rgb.clamp(0.0, 1.0)

        r, g, b_ch = rgb[:, 0], rgb[:, 1], rgb[:, 2]
        intensity = (r + g + b_ch) / 3.0
        max_rgb = torch.maximum(torch.maximum(r, g), b_ch)
        min_rgb = torch.minimum(torch.minimum(r, g), b_ch)
        saturation = (max_rgb - min_rgb).clamp_min(0.0)

        fire_cue = torch.relu(r - g) + torch.relu(r - b_ch)
        smoke_cue = (1.0 - saturation) * intensity
        color_saliency = (0.5 * fire_cue + 0.5 * smoke_cue).flatten(1)  # [B, H*W]

        # Texture: gradient magnitude
        dx = torch.zeros_like(intensity)
        dy = torch.zeros_like(intensity)
        dx[:, :, 1:] = (intensity[:, :, 1:] - intensity[:, :, :-1]).abs()
        dy[:, 1:, :] = (intensity[:, 1:, :] - intensity[:, :-1, :]).abs()
        texture = (0.5 * (dx + dy) * 0.5 + intensity * 0.5).flatten(1)  # [B, H*W]

        # Aggregate color/texture to superpixel level (mean)
        sp_color_sum = torch.zeros((B, n_segments), device=device, dtype=color_saliency.dtype)
        sp_color_sum.scatter_add_(1, safe_sp_ids, color_saliency * valid_sp_mask.float())
        sp_color_mean = sp_color_sum / sp_counts.clamp_min(1.0).to(sp_color_sum.dtype)

        sp_texture_sum = torch.zeros((B, n_segments), device=device, dtype=texture.dtype)
        sp_texture_sum.scatter_add_(1, safe_sp_ids, texture * valid_sp_mask.float())
        sp_texture_mean = sp_texture_sum / sp_counts.clamp_min(1.0).to(sp_texture_sum.dtype)

        sp_size = torch.log1p(sp_counts)

        # Normalize each component per image (vectorized)
        def _norm(vals: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
            """Min-max normalize per batch, masking invalid segments."""
            # vals: [B, N], valid: [B, N]
            big_neg = torch.finfo(vals.dtype).min
            big_pos = torch.finfo(vals.dtype).max
            masked = vals.masked_fill(~valid, big_neg)
            v_max = masked.max(dim=1, keepdim=True).values
            masked_min = vals.masked_fill(~valid, big_pos)
            v_min = masked_min.min(dim=1, keepdim=True).values
            denom = (v_max - v_min).clamp_min(1e-6)
            normed = (vals - v_min) / denom
            return normed.masked_fill(~valid, 0.0)

        sp_valid = sp_counts > 0  # [B, n_segments]
        feat_n = _norm(sp_feat_mean, sp_valid)
        color_n = _norm(sp_color_mean, sp_valid)
        texture_n = _norm(sp_texture_mean, sp_valid)
        size_n = _norm(sp_size, sp_valid)

        # Weighted combination → per-superpixel score
        sp_scores = (
            self.pixel_prune_w_feature * feat_n
            + self.pixel_prune_w_color * color_n
            + self.pixel_prune_w_texture * texture_n
            + self.pixel_prune_w_size * size_n
        )

        # Broadcast superpixel scores to pixels
        pixel_scores = sp_scores.gather(1, safe_sp_ids)  # [B, H*W]
        pixel_scores = pixel_scores.masked_fill(~pixel_valid_mask, float('-inf'))

        if debug:
            valid_scores = pixel_scores[pixel_valid_mask]
            if valid_scores.numel() > 0:
                print(f"[DETR._compute_per_pixel_scores] mode={score_mode}")
                print(f"  Pixel scores mean: {valid_scores.mean().item():.4f}, std: {valid_scores.std().item():.4f}")

        return pixel_scores

    def forward(self, samples: NestedTensor, targets=None, debug=False):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                                (center_x, center_y, height, width). These values are normalized in [0, 1],
                                relative to the size of each individual image (disregarding possible padding).
                                See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        src, mask = features[-1].decompose()
        assert mask is not None
        pixel_pos_embed = pos[-1].to(src.dtype)

        proj_src = self.input_proj(src)
        B, C, H, W = proj_src.shape
        device = proj_src.device
        P = H * W  # total pixel tokens

        pixel_src_flat = proj_src.flatten(2)         # [B, C, P]
        pixel_pos_flat = pixel_pos_embed.flatten(2)  # [B, C, P]
        pixel_mask_flat = mask.flatten(1)             # [B, P] True=padded
        pixel_valid_mask = ~pixel_mask_flat           # [B, P] True=valid

        pixel_valid_counts = pixel_valid_mask.sum(dim=1).float()  # [B]

        def _estimate_transformer_gflops(encoder_seq_len: int) -> float:
            tr = self.transformer
            d = float(tr.d_model)
            q = float(self.num_queries)
            s = float(encoder_seq_len)
            num_enc = int(getattr(tr.encoder, 'num_layers', len(getattr(tr.encoder, 'layers', []))))
            num_dec = int(getattr(tr.decoder, 'num_layers', len(getattr(tr.decoder, 'layers', []))))
            try:
                dim_ff = float(tr.encoder.layers[0].linear1.out_features)
            except Exception:
                dim_ff = 4.0 * d
            enc_self = (4.0 * s * d * d) + (2.0 * s * s * d)
            enc_ffn = 2.0 * s * d * dim_ff
            enc = num_enc * (enc_self + enc_ffn)
            dec_self = (4.0 * q * d * d) + (2.0 * q * q * d)
            dec_cross = ((2.0 * q + 2.0 * s) * d * d) + (2.0 * q * s * d)
            dec_ffn = 2.0 * q * d * dim_ff
            dec = num_dec * (dec_self + dec_cross + dec_ffn)
            return (enc + dec) / 1e9

        encoder_seq_len_before = P
        encoder_seq_len_after = P
        pixel_kept_counts = pixel_valid_counts.clone()

        if self.pixel_prune and self.hybrid_token_mode == 'mixed':
            # ── Superpixel-guided pixel token pruning ──
            # 1. Build superpixel map at feature resolution
            slic_maps = [t.get('slic_maps', {}) for t in targets] if targets is not None else []
            sp_map = self._build_pixel_superpixel_map(
                mask, slic_maps, self.slic_n_segments, H, W, device
            )
            sp_map_flat = sp_map.flatten(1)  # [B, H*W]

            # 2. Score each pixel using superpixel-guided saliency
            pixel_scores = self._compute_per_pixel_scores(
                proj_src,
                samples.tensors,
                sp_map_flat,
                pixel_valid_mask,
                self.slic_n_segments,
                self.pixel_prune_score_mode,
                debug=debug,
            )

            # 3. Determine how many tokens to keep per image
            keep_ratio = self.pixel_prune_keep_ratio
            target_keep = torch.ceil(pixel_valid_counts * keep_ratio).long()
            target_keep = torch.clamp(target_keep, min=1, max=P)

            # 4. Vectorized pruning with topk
            # Use a uniform K = max(target_keep) for batched operation, then mask
            K = int(target_keep.max().item())
            # topk on scores: returns [B, K] indices of top-scoring pixels
            _, topk_indices = pixel_scores.topk(K, dim=1, largest=True, sorted=False)

            # Gather pruned tokens
            idx_expanded = topk_indices.unsqueeze(1).expand(B, C, K)  # [B, C, K]
            pruned_src = pixel_src_flat.gather(2, idx_expanded)       # [B, C, K]
            pruned_pos = pixel_pos_flat.gather(2, idx_expanded)       # [B, C, K]

            # Build mask: tokens beyond target_keep[b] should be masked (padded)
            arange_k = torch.arange(K, device=device).unsqueeze(0)  # [1, K]
            pruned_mask = arange_k >= target_keep.unsqueeze(1)      # [B, K] True=padded

            # Also mask any token whose score was -inf (padded pixels that snuck in)
            score_at_topk = pixel_scores.gather(1, topk_indices)
            pruned_mask = pruned_mask | (score_at_topk == float('-inf'))

            # Use pruned tokens as transformer input
            transformer_src = pruned_src.unsqueeze(-1)   # [B, C, K, 1]
            transformer_pos = pruned_pos.unsqueeze(-1)   # [B, C, K, 1]
            transformer_mask = pruned_mask.unsqueeze(-1)  # [B, K, 1]

            pixel_kept_counts = (~pruned_mask).sum(dim=1).float()
            encoder_seq_len_after = K

            if debug:
                print(f"[DETR] Pixel prune: {P} -> {K} max tokens (keep_ratio={keep_ratio:.2f})")
                print(f"[DETR] Avg kept: {pixel_kept_counts.mean().item():.1f} / {pixel_valid_counts.mean().item():.1f}")

        elif self.hybrid_token_mode == 'superpixel':
            # Superpixel-only mode: pool features into superpixel tokens
            # (Legacy mode, kept for compatibility but NOT recommended for speed)
            slic_maps = [t.get('slic_maps', {}) for t in targets] if targets is not None else []
            sp_map = self._build_pixel_superpixel_map(
                mask, slic_maps, self.slic_n_segments, H, W, device
            )
            sp_map_flat = sp_map.flatten(1)  # [B, P]
            n_seg = self.slic_n_segments
            
            # Mean-pool pixel features into superpixel tokens (vectorized)
            valid_sp_mask = pixel_valid_mask & (sp_map_flat >= 0) & (sp_map_flat < n_seg)
            safe_ids = sp_map_flat.clamp(0, n_seg - 1)
            idx_exp = safe_ids.unsqueeze(1).expand(B, C, P)
            
            sp_sum = torch.zeros((B, C, n_seg), device=device, dtype=proj_src.dtype)
            sp_sum.scatter_add_(2, idx_exp, pixel_src_flat * valid_sp_mask.unsqueeze(1).float())
            
            sp_counts = torch.zeros((B, n_seg), device=device, dtype=torch.float32)
            sp_counts.scatter_add_(1, safe_ids, valid_sp_mask.float())
            
            sp_feats = sp_sum / sp_counts.unsqueeze(1).clamp_min(1.0).to(sp_sum.dtype)
            sp_mask = (sp_counts == 0)  # True = empty superpixel = padding
            
            # Position: mean centroid of pixels in each SP
            y_grid, x_grid = torch.meshgrid(
                torch.arange(H, device=device, dtype=torch.float32),
                torch.arange(W, device=device, dtype=torch.float32),
                indexing='ij',
            )
            yx = torch.stack([y_grid.flatten() / H, x_grid.flatten() / W], dim=0)  # [2, P]
            
            sp_pos_sum = torch.zeros((B, 2, n_seg), device=device, dtype=torch.float32)
            pos_idx = safe_ids.unsqueeze(1).expand(B, 2, P)
            sp_pos_sum.scatter_add_(2, pos_idx, yx.unsqueeze(0).expand(B, -1, -1) * valid_sp_mask.unsqueeze(1).float())
            sp_centroids = sp_pos_sum / sp_counts.unsqueeze(1).clamp_min(1.0)

            # Generate position embeddings for SP centroids using sine encoding
            from .position_encoding import PositionEmbeddingSine
            # Quick sine pos embed for centroids
            scale = 2 * 3.14159265358979
            num_pos_feats = C // 2
            dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=device)
            dim_t = 10000.0 ** (2 * (dim_t // 2) / num_pos_feats)
            
            y_embed = sp_centroids[:, 0:1, :] * scale  # [B, 1, n_seg]
            x_embed = sp_centroids[:, 1:2, :] * scale  # [B, 1, n_seg]
            
            pos_y = y_embed.unsqueeze(-1) / dim_t  # [B, 1, n_seg, num_pos_feats]
            pos_x = x_embed.unsqueeze(-1) / dim_t
            pos_y = torch.stack([pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()], dim=-1).flatten(-2)
            pos_x = torch.stack([pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()], dim=-1).flatten(-2)
            sp_pos = torch.cat([pos_y, pos_x], dim=-1).squeeze(1).permute(0, 2, 1)  # [B, C, n_seg]

            transformer_src = sp_feats.unsqueeze(-1)     # [B, C, n_seg, 1]
            transformer_pos = sp_pos.unsqueeze(-1)        # [B, C, n_seg, 1]
            transformer_mask = sp_mask.unsqueeze(-1)      # [B, n_seg, 1]

            encoder_seq_len_before = n_seg
            encoder_seq_len_after = n_seg
            pixel_kept_counts = torch.zeros((B,), device=device, dtype=torch.float32)

        else:
            # No pruning — vanilla pixel tokens (fallback / default)
            transformer_src = proj_src.unsqueeze(-1) if proj_src.dim() == 3 else proj_src
            # Ensure 4D shape [B, C, P, 1] for transformer
            if transformer_src.dim() == 4 and transformer_src.shape[-1] != 1:
                transformer_src = proj_src.flatten(2).unsqueeze(-1)
                transformer_pos = pixel_pos_embed.flatten(2).unsqueeze(-1)
                transformer_mask = pixel_mask_flat.unsqueeze(-1)
            else:
                transformer_src = proj_src
                transformer_pos = pixel_pos_embed
                transformer_mask = mask

        hs = self.transformer(
            transformer_src,
            transformer_mask,
            self.query_embed.weight,
            transformer_pos,
            debug=debug,
        )[0]

        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        out = {
            'pred_logits': outputs_class[-1],
            'pred_boxes': outputs_coord[-1],
        }

        # Efficiency metrics
        encoder_tokens_valid_before = pixel_valid_counts
        encoder_tokens_valid_after = pixel_kept_counts

        denom_pix = float(pixel_valid_counts.sum().clamp_min(1.0).item())
        pix_keep_ratio_actual = float(pixel_kept_counts.sum().item() / denom_pix)

        seq_ratio = float(encoder_seq_len_after / max(encoder_seq_len_before, 1))
        gflops_before = _estimate_transformer_gflops(encoder_seq_len_before)
        gflops_after = _estimate_transformer_gflops(encoder_seq_len_after)
        gflops_ratio = float(gflops_after / max(gflops_before, 1e-12))

        out.update({
            'eff_pixel_prune_enabled': int(self.pixel_prune and self.hybrid_token_mode == 'mixed'),
            'eff_pixel_prune_keep_ratio_target': float(self.pixel_prune_keep_ratio),
            'eff_pixel_keep_ratio_actual': pix_keep_ratio_actual,
            'eff_pixel_tokens_before': float(pixel_valid_counts.mean().item()),
            'eff_pixel_tokens_after': float(pixel_kept_counts.mean().item()),
            'eff_superpixel_tokens': 0.0,
            'eff_tokens_before': float(encoder_tokens_valid_before.mean().item()),
            'eff_tokens_after': float(encoder_tokens_valid_after.mean().item()),
            'eff_tokens_ratio': float((encoder_tokens_valid_after.sum() / encoder_tokens_valid_before.sum().clamp_min(1.0)).item()),
            'eff_encoder_seq_len_before': float(encoder_seq_len_before),
            'eff_encoder_seq_len_after': float(encoder_seq_len_after),
            'eff_encoder_seq_len_ratio': seq_ratio,
            'eff_encoder_seq_len_reduction': float(1.0 - seq_ratio),
            'eff_gflops_before': float(gflops_before),
            'eff_gflops_after': float(gflops_after),
            'eff_gflops_ratio': gflops_ratio,
        })

        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)

        if debug:
            print(f"[DETR] Hybrid token mode: {self.hybrid_token_mode}")
            print(f"[DETR] Pixel prune enabled: {self.pixel_prune}")
            if self.pixel_prune:
                print(f"[DETR] Pixel prune keep ratio target: {self.pixel_prune_keep_ratio}")
                print(f"[DETR] Pixel prune score mode: {self.pixel_prune_score_mode}")
            print(f"[DETR] Input samples tensor shape: {samples.tensors.shape}")
            print(f"[DETR] Transformer src shape: {transformer_src.shape}")
            print(f"[DETR] Output pred_logits shape: {out['pred_logits'].shape}")
            print(f"[DETR] Output pred_boxes shape: {out['pred_boxes'].shape}")
            print(f"[DETR] eff_encoder_seq_len_before: {encoder_seq_len_before}")
            print(f"[DETR] eff_encoder_seq_len_after: {encoder_seq_len_after}")
            print(f"[DETR] eff_pixel_tokens_before(avg): {out['eff_pixel_tokens_before']:.1f}")
            print(f"[DETR] eff_pixel_tokens_after(avg): {out['eff_pixel_tokens_after']:.1f}")
            print(f"[DETR] eff_gflops_before: {out['eff_gflops_before']:.3f}")
            print(f"[DETR] eff_gflops_after: {out['eff_gflops_after']:.3f}")

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)
        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss in ('masks',):
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = F.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)

        # convert to [x0, y0, x1, y1] format
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        return results


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args):
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
    # is the maximum id for a class in your dataset. For example,
    # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    # As another example, for a dataset that has a single class with id 1,
    # you should pass `num_classes` to be 2 (max_obj_id + 1).
    # For more details on this, check the following discussion
    # https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223
    num_classes = 2
    if args.dataset_file == "coco_panoptic":
        # for panoptic, we just add a num_classes that is large enough to hold
        # max_obj_id + 1, but the exact value doesn't really matter
        num_classes = 250
    device = torch.device(args.device)

    backbone = build_backbone(args)

    transformer = build_transformer(args)

    model = DETR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
        slic_n_segments=getattr(args, 'slic_n_segments', 200),
        hybrid_token_mode=getattr(args, 'hybrid_token_mode', 'mixed'),
        pixel_prune=getattr(args, 'pixel_prune', False),
        pixel_prune_keep_ratio=getattr(args, 'pixel_prune_keep_ratio', 0.8),
        pixel_prune_score_mode=getattr(args, 'pixel_prune_score_mode', 'saliency'),
        pixel_prune_w_feature=getattr(args, 'pixel_prune_w_feature', 0.45),
        pixel_prune_w_color=getattr(args, 'pixel_prune_w_color', 0.25),
        pixel_prune_w_texture=getattr(args, 'pixel_prune_w_texture', 0.20),
        pixel_prune_w_size=getattr(args, 'pixel_prune_w_size', 0.10),
    )
    if args.masks:
        model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))
    matcher = build_matcher(args)
    weight_dict = {'loss_ce': 1, 'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef
    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'cardinality']
    if args.masks:
        losses += ["masks"]
    criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
                             eos_coef=args.eos_coef, losses=losses)
    criterion.to(device)
    postprocessors = {'bbox': PostProcess()}
    if args.masks:
        postprocessors['segm'] = PostProcessSegm()
        if args.dataset_file == "coco_panoptic":
            is_thing_map = {i: i <= 90 for i in range(201)}
            postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)

    return model, criterion, postprocessors
