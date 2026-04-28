# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn

try:
    from torch_scatter import scatter_add, scatter_max
    HAS_TORCH_SCATTER = True
except Exception:
    scatter_add = None
    scatter_max = None
    HAS_TORCH_SCATTER = False

from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)

from .backbone import build_backbone
from .matcher import build_matcher
from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
                           dice_loss, sigmoid_focal_loss)
from .position_encoding import PositionEmbeddingSuperpixel
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
        pooling_type='mean',
        hybrid_token_mode='mixed',
        compact_superpixel_ids=False,
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
        self.pooling_type = pooling_type
        self.hybrid_token_mode = hybrid_token_mode
        self.compact_superpixel_ids = compact_superpixel_ids
        self.pixel_prune = pixel_prune
        self.pixel_prune_keep_ratio = float(pixel_prune_keep_ratio)
        self.pixel_prune_keep_ratio = max(0.6, min(0.8, self.pixel_prune_keep_ratio))
        self.pixel_prune_score_mode = pixel_prune_score_mode
        self.pixel_prune_w_feature = pixel_prune_w_feature
        self.pixel_prune_w_color = pixel_prune_w_color
        self.pixel_prune_w_texture = pixel_prune_w_texture
        self.pixel_prune_w_size = pixel_prune_w_size
        self.pos_embed_superpixel = PositionEmbeddingSuperpixel(hidden_dim // 2)
        self.register_buffer(
            'superpixel_token_ids',
            torch.arange(slic_n_segments, dtype=torch.long),
            persistent=False,
        )
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

    def _compact_superpixel_ids(self, s_map: torch.Tensor, n_segments: int) -> torch.Tensor:
        compacted = torch.full_like(s_map, -1)
        valid_mask = s_map >= 0
        if not valid_mask.any():
            return compacted

        valid_labels = s_map[valid_mask]
        unique_ids, counts = torch.unique(valid_labels, return_counts=True)
        keep_count = min(n_segments, unique_ids.numel())
        sorted_idx = torch.argsort(counts, descending=True)
        kept_ids = unique_ids[sorted_idx[:keep_count]]

        sorted_kept_ids, sorted_pos = torch.sort(kept_ids)
        search_pos = torch.searchsorted(sorted_kept_ids, valid_labels)
        safe_pos = search_pos.clamp(max=max(keep_count - 1, 0))
        matched = (search_pos < keep_count) & (sorted_kept_ids[safe_pos] == valid_labels)

        mapped = torch.full_like(valid_labels, -1)
        mapped[matched] = sorted_pos[search_pos[matched]]
        compacted[valid_mask] = mapped

        return compacted

    def pool_superpixel_features(
        self,
        features,
        mask,
        slic_maps,
        n_segments,
        pooling_type='mean',
        compact_ids=False,
        debug=False,
    ):
        B, C, H, W = features.shape
        device = features.device
        
        pooled_features = torch.zeros((B, C, n_segments), device=device, dtype=features.dtype)
        pooled_mask = torch.ones((B, n_segments), device=device, dtype=torch.bool)
        pooled_pos = torch.zeros((B, 2, n_segments), device=device, dtype=torch.float32) # y, x
        pooled_counts = torch.zeros((B, n_segments), device=device, dtype=torch.float32)
        
        # Batch resize slic_maps to feature size [B, H, W]
        batched_slic_down = torch.full((B, H, W), -1, device=device, dtype=torch.long)
        
        for b in range(B):
            # Calculate valid spatial region from the backbone mask.
            valid_y_mask_b = (~mask[b]).any(dim=1) # [H]
            valid_x_mask_b = (~mask[b]).any(dim=0) # [W]
            valid_H = valid_y_mask_b.sum().item()
            valid_W = valid_x_mask_b.sum().item()
            
            if valid_H == 0 or valid_W == 0:
                continue
                
            s_map = None
            if slic_maps and b < len(slic_maps):
                s_map = slic_maps[b].get(n_segments, None)
                
            if s_map is None:
                # Default to processing with segment 0 if none provided
                batched_slic_down[b, :valid_H, :valid_W] = 0
            else:
                # Downsample the single image's slic_map to its exact validity scale.
                s_map_down = F.interpolate(
                    s_map[None, None].float(),
                    size=(valid_H, valid_W),
                    mode='nearest'
                )[0, 0].long()

                if s_map_down.device != device:
                    s_map_down = s_map_down.to(device=device, non_blocking=True)

                if compact_ids:
                    s_map_down = self._compact_superpixel_ids(s_map_down, n_segments)
                else:
                    invalid_mask = (s_map_down < 0) | (s_map_down >= n_segments)
                    s_map_down[invalid_mask] = -1

                batched_slic_down[b, :valid_H, :valid_W] = s_map_down
        
        feat_flat = features.flatten(2) # [B, C, H*W]
        slic_flat = batched_slic_down.flatten(1) # [B, H*W]
        
        y_grid, x_grid = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
        y_grid = y_grid.flatten().float() # [H*W]
        x_grid = x_grid.flatten().float() # [H*W]
        
        for b in range(B):
            sp_ids = slic_flat[b]
            valid_token_mask = (sp_ids >= 0) & (sp_ids < n_segments)
            if not valid_token_mask.any():
                continue

            valid_sp_ids = sp_ids[valid_token_mask]

            if HAS_TORCH_SCATTER:
                valid_feats = feat_flat[b, :, valid_token_mask]  # [C, num_valid_tokens]
                feat_index = valid_sp_ids.unsqueeze(0).expand(C, -1)

                counts = scatter_add(
                    torch.ones_like(valid_sp_ids, dtype=torch.float32),
                    valid_sp_ids,
                    dim=0,
                    dim_size=n_segments,
                )
                counts_for_feat = counts.to(valid_feats.dtype).clamp_min(1.0).unsqueeze(0)

                mean_feats = scatter_add(valid_feats, feat_index, dim=1, dim_size=n_segments) / counts_for_feat
                max_feats = None
                if pooling_type != 'mean':
                    max_feats, _ = scatter_max(valid_feats, feat_index, dim=1, dim_size=n_segments)
                    max_feats = max_feats.masked_fill(counts.unsqueeze(0) == 0, 0.0)

                if pooling_type == 'mean':
                    pooled = mean_feats
                elif pooling_type == 'max':
                    pooled = max_feats
                else:
                    pooled = (mean_feats + max_feats) / 2.0

                valid_y = y_grid[valid_token_mask]
                valid_x = x_grid[valid_token_mask]
                y_sum = scatter_add(valid_y, valid_sp_ids, dim=0, dim_size=n_segments)
                x_sum = scatter_add(valid_x, valid_sp_ids, dim=0, dim_size=n_segments)

                valid_segment_mask = counts > 0
                pooled_features[b] = pooled
                pooled_mask[b] = ~valid_segment_mask
                pooled_counts[b] = counts
                pooled_pos[b, 0, valid_segment_mask] = (y_sum[valid_segment_mask] / counts[valid_segment_mask]) / H
                pooled_pos[b, 1, valid_segment_mask] = (x_sum[valid_segment_mask] / counts[valid_segment_mask]) / W
            else:
                unique_sps = torch.unique(valid_sp_ids)

                for sp_id_tensor in unique_sps:
                    sp_id = int(sp_id_tensor.item())
                    sp_mask = (sp_ids == sp_id) # [H*W]
                    if not sp_mask.any():
                        continue

                    sp_feats = feat_flat[b, :, sp_mask] # [C, num_pixels]
                    if pooling_type == 'mean':
                        pooled = sp_feats.mean(dim=1)
                    elif pooling_type == 'max':
                        pooled = sp_feats.max(dim=1)[0]
                    else:
                        pooled = (sp_feats.mean(dim=1) + sp_feats.max(dim=1)[0]) / 2.0

                    pooled_features[b, :, sp_id] = pooled
                    pooled_mask[b, sp_id] = False
                    pooled_counts[b, sp_id] = float(sp_mask.sum().item())

                    valid_y = y_grid[sp_mask]
                    valid_x = x_grid[sp_mask]
                    pooled_pos[b, 0, sp_id] = valid_y.mean() / H
                    pooled_pos[b, 1, sp_id] = valid_x.mean() / W

        if debug:
            print(f"[DETR.pool_superpixel_features] torch_scatter enabled: {HAS_TORCH_SCATTER}")
            print(f"[DETR.pool_superpixel_features] Input features shape: {features.shape}")
            print(f"[DETR.pool_superpixel_features] Input features mean: {features.mean().item():.4f}, std: {features.std().item():.4f}")
            print(f"[DETR.pool_superpixel_features] Output pooled_features shape: {pooled_features.shape}")
            print(f"[DETR.pool_superpixel_features] Output pooled_features mean: {pooled_features.mean().item():.4f}, std: {pooled_features.std().item():.4f}")
                
        return pooled_features, pooled_mask, pooled_pos, pooled_counts, batched_slic_down

    def _normalize_component_per_image(self, values: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        normalized = torch.zeros_like(values)
        eps = 1e-6
        for b in range(values.shape[0]):
            valid = valid_mask[b]
            if not valid.any():
                continue
            v = values[b, valid]
            v_min = v.min()
            v_max = v.max()
            if (v_max - v_min) > eps:
                normalized[b, valid] = (v - v_min) / (v_max - v_min)
            else:
                normalized[b, valid] = 0.5
        return normalized

    def _aggregate_superpixel_scalar(
        self,
        scalar_map: torch.Tensor,
        batched_slic_down: torch.Tensor,
        pooled_counts: torch.Tensor,
    ) -> torch.Tensor:
        B, n_segments = pooled_counts.shape
        out = torch.zeros((B, n_segments), device=scalar_map.device, dtype=scalar_map.dtype)

        for b in range(B):
            sp_ids = batched_slic_down[b].reshape(-1)
            vals = scalar_map[b].reshape(-1)
            valid = (sp_ids >= 0) & (sp_ids < n_segments)
            if not valid.any():
                continue

            valid_sp_ids = sp_ids[valid]
            valid_vals = vals[valid]

            if HAS_TORCH_SCATTER:
                sums = scatter_add(valid_vals, valid_sp_ids, dim=0, dim_size=n_segments)
            else:
                sums = torch.bincount(valid_sp_ids, weights=valid_vals, minlength=n_segments)
                sums = sums.to(device=scalar_map.device, dtype=scalar_map.dtype)

            out[b] = sums / pooled_counts[b].to(scalar_map.dtype).clamp_min(1.0)

        return out

    def compute_superpixel_prune_scores(
        self,
        samples_tensors: torch.Tensor,
        pooled_src: torch.Tensor,
        pooled_mask: torch.Tensor,
        pooled_counts: torch.Tensor,
        batched_slic_down: torch.Tensor,
        score_mode: str = 'saliency',
        debug: bool = False,
    ) -> torch.Tensor:
        """Compute per-superpixel ranking scores used for pixel-token pruning.

        Returns:
            scores: [B, n_segments] with invalid segments filled with -1.
        """
        valid_segments = (~pooled_mask) & (pooled_counts > 0)

        if score_mode == 'counts':
            scores = pooled_counts.to(torch.float32)
            return scores.masked_fill(~valid_segments, -1.0)

        if score_mode == 'feature_norm':
            scores = torch.linalg.vector_norm(pooled_src, ord=2, dim=1).to(torch.float32)
            return scores.masked_fill(~valid_segments, -1.0)

        if score_mode != 'saliency':
            raise ValueError(f"Unknown pixel_prune_score_mode: {score_mode}")

        _, Hf, Wf = batched_slic_down.shape

        rgb = samples_tensors
        if rgb.shape[-2:] != (Hf, Wf):
            rgb = F.interpolate(rgb, size=(Hf, Wf), mode='bilinear', align_corners=False)
        rgb = (rgb * self.input_std.to(rgb.dtype)) + self.input_mean.to(rgb.dtype)
        rgb = rgb.clamp(0.0, 1.0)

        r = rgb[:, 0]
        g = rgb[:, 1]
        b = rgb[:, 2]
        intensity = (r + g + b) / 3.0
        max_rgb = torch.maximum(torch.maximum(r, g), b)
        min_rgb = torch.minimum(torch.minimum(r, g), b)
        saturation = (max_rgb - min_rgb).clamp_min(0.0)

        fire_cue = torch.relu(r - g) + torch.relu(r - b)
        smoke_cue = (1.0 - saturation) * intensity
        color_saliency_map = 0.5 * fire_cue + 0.5 * smoke_cue

        dx = torch.zeros_like(intensity)
        dy = torch.zeros_like(intensity)
        dx[:, :, 1:] = (intensity[:, :, 1:] - intensity[:, :, :-1]).abs()
        dy[:, 1:, :] = (intensity[:, 1:, :] - intensity[:, :-1, :]).abs()
        texture_map = 0.5 * (dx + dy)
        texture_intensity_map = 0.5 * texture_map + 0.5 * intensity

        feature_norm = torch.linalg.vector_norm(pooled_src, ord=2, dim=1)
        color_saliency = self._aggregate_superpixel_scalar(color_saliency_map, batched_slic_down, pooled_counts)
        texture_intensity = self._aggregate_superpixel_scalar(texture_intensity_map, batched_slic_down, pooled_counts)
        size_prior = torch.log1p(pooled_counts)

        feature_norm_n = self._normalize_component_per_image(feature_norm, valid_segments)
        color_saliency_n = self._normalize_component_per_image(color_saliency, valid_segments)
        texture_intensity_n = self._normalize_component_per_image(texture_intensity, valid_segments)
        size_prior_n = self._normalize_component_per_image(size_prior, valid_segments)

        scores = (
            self.pixel_prune_w_feature * feature_norm_n
            + self.pixel_prune_w_color * color_saliency_n
            + self.pixel_prune_w_texture * texture_intensity_n
            + self.pixel_prune_w_size * size_prior_n
        )
        scores = scores.to(torch.float32).masked_fill(~valid_segments, -1.0)

        if debug:
            print(
                "[DETR.compute_superpixel_prune_scores] "
                f"mode={score_mode} w(feature/color/texture/size)="
                f"{self.pixel_prune_w_feature}/{self.pixel_prune_w_color}/{self.pixel_prune_w_texture}/{self.pixel_prune_w_size}"
            )
            print(f"[DETR.compute_superpixel_prune_scores] Output scores shape: {scores.shape}")
            valid_vals = scores[valid_segments]
            if valid_vals.numel() > 0:
                print(f"[DETR.compute_superpixel_prune_scores] Scores mean: {valid_vals.mean().item():.4f}, std: {valid_vals.std().item():.4f}")

        return scores

    def forward(self, samples: NestedTensor, targets=None, debug=False):
        """ The forward expects a NestedTensor, which consists of:
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
        
        slic_maps = [t.get('slic_maps', {}) for t in targets] if targets is not None else []
        pooled_src, pooled_mask, pooled_pos, pooled_counts, batched_slic_down = self.pool_superpixel_features(
            proj_src,
            mask,
            slic_maps,
            self.slic_n_segments,
            self.pooling_type,
            compact_ids=self.compact_superpixel_ids,
            debug=debug,
        )
        
        # Sine encoding for superpixel centroids
        # pooled_pos: [B, 2, N] -> [B, 2, N, 1]
        pooled_pos_2d = pooled_pos.unsqueeze(-1)
        superpixel_pos_embed = self.pos_embed_superpixel(pooled_pos_2d, debug=debug) # [B, 256, N, 1]

        def _estimate_transformer_gflops(encoder_seq_len: int) -> float:
            # Rough analytic estimate per image forward pass.
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

            # Self-attention (approx): 4*L*d^2 + 2*L^2*d
            # FFN (approx): 2*L*d*dim_ff
            enc_self = (4.0 * s * d * d) + (2.0 * s * s * d)
            enc_ffn = 2.0 * s * d * dim_ff
            enc = num_enc * (enc_self + enc_ffn)

            dec_self = (4.0 * q * d * d) + (2.0 * q * q * d)
            dec_cross = ((2.0 * q + 2.0 * s) * d * d) + (2.0 * q * s * d)
            dec_ffn = 2.0 * q * d * dim_ff
            dec = num_dec * (dec_self + dec_cross + dec_ffn)

            return (enc + dec) / 1e9

        B = proj_src.shape[0]
        device = proj_src.device
        slic_n_segments = pooled_src.shape[2]

        # Defaults for efficiency metrics
        pixel_valid_counts = torch.zeros((B,), device=device, dtype=torch.float32)
        pixel_kept_counts = torch.zeros((B,), device=device, dtype=torch.float32)
        superpixel_valid_counts = (~pooled_mask).sum(dim=1).to(torch.float32)

        if self.hybrid_token_mode == 'mixed':
            pixel_src_tokens = proj_src.flatten(2)           # [B, C, P]
            pixel_pos_tokens = pixel_pos_embed.flatten(2)    # [B, C, P]
            pixel_mask_tokens = mask.flatten(1)              # [B, P] (True = padded)
            pixel_sp_tokens = batched_slic_down.flatten(1)   # [B, P]

            superpixel_src_tokens = pooled_src
            superpixel_pos_tokens = superpixel_pos_embed.squeeze(-1)  # [B, C, N]
            superpixel_mask_tokens = pooled_mask

            pixel_valid_mask = ~pixel_mask_tokens
            pixel_valid_counts = pixel_valid_mask.sum(dim=1).to(torch.float32)
            pixel_kept_counts = pixel_valid_counts

            P = int(pixel_src_tokens.shape[2])
            encoder_seq_len_before = P + int(slic_n_segments)
            encoder_seq_len_after = encoder_seq_len_before

            if self.pixel_prune:
                keep_ratio = float(self.pixel_prune_keep_ratio)

                scores = self.compute_superpixel_prune_scores(
                    samples.tensors,
                    pooled_src,
                    pooled_mask,
                    pooled_counts,
                    batched_slic_down,
                    score_mode=self.pixel_prune_score_mode,
                    debug=debug,
                )

                valid_sp_pixel_mask = pixel_valid_mask & (pixel_sp_tokens >= 0) & (pixel_sp_tokens < slic_n_segments)
                invalid_sp_pixel_mask = pixel_valid_mask & (~valid_sp_pixel_mask)

                target_keep = torch.ceil(pixel_valid_counts * keep_ratio).to(torch.long)
                target_keep = torch.minimum(target_keep, pixel_valid_counts.to(torch.long))
                invalid_keep = invalid_sp_pixel_mask.sum(dim=1).to(torch.long)
                remaining_keep = (target_keep - invalid_keep).clamp(min=0)

                sorted_scores, sorted_idx = scores.sort(dim=1, descending=True)
                sorted_counts = pooled_counts.gather(1, sorted_idx).to(torch.float32)
                sorted_counts = sorted_counts * (sorted_scores >= 0).to(sorted_counts.dtype)

                cum_counts = sorted_counts.cumsum(dim=1)
                need = remaining_keep.to(cum_counts.dtype).unsqueeze(1)
                below = cum_counts < need
                k = below.sum(dim=1) + (remaining_keep > 0).to(torch.long)
                k = k.clamp(max=int(slic_n_segments))

                arange = torch.arange(int(slic_n_segments), device=device).unsqueeze(0).expand(B, -1)
                select_sorted = arange < k.unsqueeze(1)
                keep_sp_mask = torch.zeros((B, int(slic_n_segments)), device=device, dtype=torch.bool)
                keep_sp_mask.scatter_(1, sorted_idx, select_sorted)

                safe_sp_ids = pixel_sp_tokens.clamp(min=0, max=max(int(slic_n_segments) - 1, 0))
                keep_from_sp = keep_sp_mask.gather(1, safe_sp_ids)
                pixel_keep_mask = (valid_sp_pixel_mask & keep_from_sp) | invalid_sp_pixel_mask

                # Gather variable-length kept pixel tokens and pad to max kept length in the batch.
                kept_lens = []
                kept_indices = []
                for b in range(B):
                    idx = torch.nonzero(pixel_keep_mask[b], as_tuple=False).flatten()
                    kept_indices.append(idx)
                    kept_lens.append(int(idx.numel()))

                max_kept = max(kept_lens) if kept_lens else 0
                pruned_src = pixel_src_tokens.new_zeros((B, pixel_src_tokens.shape[1], max_kept))
                pruned_pos = pixel_pos_tokens.new_zeros((B, pixel_pos_tokens.shape[1], max_kept))
                pruned_mask = torch.ones((B, max_kept), device=device, dtype=torch.bool)

                for b in range(B):
                    m = kept_lens[b]
                    if m == 0:
                        continue
                    idx = kept_indices[b]
                    pruned_src[b, :, :m] = pixel_src_tokens[b, :, idx]
                    pruned_pos[b, :, :m] = pixel_pos_tokens[b, :, idx]
                    pruned_mask[b, :m] = False

                pixel_src_tokens = pruned_src
                pixel_pos_tokens = pruned_pos
                pixel_mask_tokens = pruned_mask

                pixel_kept_counts = pixel_keep_mask.sum(dim=1).to(torch.float32)

                encoder_seq_len_after = int(max_kept) + int(slic_n_segments)

            transformer_src = torch.cat([pixel_src_tokens, superpixel_src_tokens], dim=2).unsqueeze(-1)
            transformer_pos = torch.cat([pixel_pos_tokens, superpixel_pos_tokens], dim=2).unsqueeze(-1)
            transformer_mask = torch.cat([pixel_mask_tokens, superpixel_mask_tokens], dim=1).unsqueeze(-1)
        else:
            encoder_seq_len_before = int(slic_n_segments)
            encoder_seq_len_after = int(slic_n_segments)
            transformer_src = pooled_src.unsqueeze(-1)
            transformer_pos = superpixel_pos_embed
            transformer_mask = pooled_mask.unsqueeze(-1)

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

        # Efficiency metrics (token counts are averages per image for this batch).
        encoder_tokens_valid_before = pixel_valid_counts + superpixel_valid_counts
        encoder_tokens_valid_after = pixel_kept_counts + superpixel_valid_counts

        denom_pix = float(pixel_valid_counts.sum().clamp_min(1.0).item())
        pix_keep_ratio_actual = float(pixel_kept_counts.sum().item() / denom_pix)

        seq_ratio = float(encoder_seq_len_after / max(encoder_seq_len_before, 1))
        gflops_before = _estimate_transformer_gflops(int(encoder_seq_len_before))
        gflops_after = _estimate_transformer_gflops(int(encoder_seq_len_after))
        gflops_ratio = float(gflops_after / max(gflops_before, 1e-12))

        out.update({
            'eff_pixel_prune_enabled': int(self.pixel_prune and self.hybrid_token_mode == 'mixed'),
            'eff_pixel_prune_keep_ratio_target': float(self.pixel_prune_keep_ratio),
            'eff_pixel_keep_ratio_actual': pix_keep_ratio_actual,
            'eff_pixel_tokens_before': float(pixel_valid_counts.mean().item()),
            'eff_pixel_tokens_after': float(pixel_kept_counts.mean().item()),
            'eff_superpixel_tokens': float(superpixel_valid_counts.mean().item()),
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
            print(f"[DETR] Input samples tensor mean: {samples.tensors.mean().item():.4f}, std: {samples.tensors.std().item():.4f}")
            print(f"[DETR] Transformer src shape: {transformer_src.shape}")
            print(f"[DETR] Transformer src mean: {transformer_src.mean().item():.4f}, std: {transformer_src.std().item():.4f}")
            print(f"[DETR] Output pred_logits shape: {out['pred_logits'].shape}")
            print(f"[DETR] Output pred_boxes shape: {out['pred_boxes'].shape}")
            print(f"[DETR] Output pred_logits mean: {out['pred_logits'].mean().item():.4f}, std: {out['pred_logits'].std().item():.4f}")
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
        pooling_type=getattr(args, 'pooling_type', 'mean'),
        hybrid_token_mode=getattr(args, 'hybrid_token_mode', 'mixed'),
        compact_superpixel_ids=getattr(args, 'compact_superpixel_ids', False),
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
