# ------------------------------------------------------------------------
# DINO-SLIC Transformer
# Standard MHSA Transformer for superpixel token sequences.
# Replaces DeformableTransformer (which requires spatial grids).
# ------------------------------------------------------------------------

"""
SLICTransformer: A transformer encoder-decoder for superpixel tokens.

Unlike DeformableTransformer which relies on spatial grids and deformable
attention (bilinear sampling at spatial offsets), this transformer uses
standard multi-head self-attention and cross-attention, which naturally
operate on any set of tokens with positional encodings.

With ~700 superpixel tokens (vs ~8400 grid cells), standard O(n²) attention
is both feasible and efficient.
"""

import math
import copy
import random
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from models.dino.utils import (
    gen_sineembed_for_position,
    MLP,
    _get_activation_fn,
)
from util.misc import inverse_sigmoid


# =============================================================================
# Positional Encoding for Superpixel Centroids
# =============================================================================
class SuperpixelPositionEmbedding(nn.Module):
    """Sinusoidal positional encoding from normalized (cx, cy) centroids.

    Uses the same sine/cosine formula as DINO's PositionEmbeddingSineHW,
    but operates on arbitrary 2D coordinates instead of grid positions.

    Args:
        num_pos_feats: half of d_model (128 for d_model=256)
        temperature: temperature for sinusoidal encoding
    """

    def __init__(self, num_pos_feats: int = 128, temperature: float = 10000.0):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.scale = 2 * math.pi

    def forward(self, centroids: Tensor) -> Tensor:
        """
        Args:
            centroids: [bs, N, 2] normalized (cx, cy) in [0, 1]
        Returns:
            pos: [bs, N, d_model] where d_model = 2 * num_pos_feats
        """
        # Scale to [0, 2π]
        cx = centroids[:, :, 0:1] * self.scale  # [bs, N, 1]
        cy = centroids[:, :, 1:2] * self.scale  # [bs, N, 1]

        dim_t = torch.arange(
            self.num_pos_feats, dtype=torch.float32, device=centroids.device
        )
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        # Encode x: [bs, N, num_pos_feats]
        pos_x = cx / dim_t  # [bs, N, num_pos_feats]
        pos_x = torch.stack(
            (pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3
        ).flatten(2)  # [bs, N, num_pos_feats]

        # Encode y: [bs, N, num_pos_feats]
        pos_y = cy / dim_t
        pos_y = torch.stack(
            (pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3
        ).flatten(2)

        # Concatenate: [bs, N, d_model]
        pos = torch.cat((pos_y, pos_x), dim=2)
        return pos


# =============================================================================
# Encoder Layer
# =============================================================================
class SLICEncoderLayer(nn.Module):
    """Standard transformer encoder layer: MHSA + FFN.

    Args:
        d_model: feature dimension
        nhead: number of attention heads
        dim_feedforward: FFN hidden dimension
        dropout: dropout rate
        activation: activation function name
    """

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.0,
        activation: str = "relu",
        debug: bool = False,
    ):
        super().__init__()
        self.debug = debug

        # Self-attention
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # FFN
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor: Tensor, pos: Optional[Tensor]) -> Tensor:
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        src: Tensor,                              # [N, bs, d_model]
        pos: Optional[Tensor] = None,             # [N, bs, d_model]
        key_padding_mask: Optional[Tensor] = None, # [bs, N]
    ) -> Tensor:
        # Self-attention
        q = k = self.with_pos_embed(src, pos)
        src2, _ = self.self_attn(q, k, src, key_padding_mask=key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # FFN
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)

        if self.debug:
            print(f"[SLICEncoderLayer] src: {src.shape}, mean={src.mean():.4f}, std={src.std():.4f}")

        return src


# =============================================================================
# Encoder
# =============================================================================
class SLICEncoder(nn.Module):
    """Stack of SLICEncoderLayer.

    Args:
        encoder_layer: a single encoder layer (will be deep-copied)
        num_layers: number of encoder layers
        norm: optional final layer norm
        d_model: feature dimension
    """

    def __init__(
        self,
        encoder_layer: nn.Module,
        num_layers: int,
        norm: Optional[nn.Module] = None,
        d_model: int = 256,
    ):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.norm = norm
        self.d_model = d_model
        self.num_layers = num_layers
        # Learnable residual connection from v1
        self.residual_weight = nn.Parameter(torch.tensor(-1.0))
        self.final_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        src: Tensor,                              # [N, bs, d_model]
        pos: Optional[Tensor] = None,             # [N, bs, d_model]
        key_padding_mask: Optional[Tensor] = None, # [bs, N]
    ) -> Tensor:
        output = src
        for layer in self.layers:
            output = layer(output, pos=pos, key_padding_mask=key_padding_mask)
        if self.norm is not None:
            output = self.norm(output)

        # Learnable residual mixing: alpha*encoded + (1-alpha)*src
        alpha = torch.sigmoid(self.residual_weight)
        output = alpha * output + (1 - alpha) * src
        output = self.final_norm(output)

        return output


# =============================================================================
# Decoder Layer
# =============================================================================
class SLICDecoderLayer(nn.Module):
    """Transformer decoder layer: self-attn + cross-attn + FFN.

    Uses standard nn.MultiheadAttention for both self-attention (among queries)
    and cross-attention (queries attend to superpixel memory).

    Args:
        d_model: feature dimension
        nhead: number of attention heads
        dim_feedforward: FFN hidden dimension
        dropout: dropout rate
        activation: activation function name
    """

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.0,
        activation: str = "relu",
        debug: bool = False,
    ):
        super().__init__()
        self.debug = debug

        # Self-attention (among object queries)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # Cross-attention (queries → superpixel memory)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # FFN
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor: Tensor, pos: Optional[Tensor]) -> Tensor:
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        tgt: Tensor,                                    # [nq, bs, d_model]
        tgt_query_pos: Optional[Tensor] = None,          # [nq, bs, d_model]
        memory: Optional[Tensor] = None,                 # [N_mem, bs, d_model]
        memory_key_padding_mask: Optional[Tensor] = None, # [bs, N_mem]
        memory_pos: Optional[Tensor] = None,             # [N_mem, bs, d_model]
        self_attn_mask: Optional[Tensor] = None,         # [nq, nq]
    ) -> Tensor:
        # Self-attention
        q = k = self.with_pos_embed(tgt, tgt_query_pos)
        tgt2, _ = self.self_attn(q, k, tgt, attn_mask=self_attn_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # Cross-attention
        q = self.with_pos_embed(tgt, tgt_query_pos)
        k = self.with_pos_embed(memory, memory_pos)
        tgt2, _ = self.cross_attn(q, k, memory, key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # FFN
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)

        if self.debug:
            print(f"[SLICDecoderLayer] tgt: {tgt.shape}, mean={tgt.mean():.4f}")

        return tgt


# =============================================================================
# Decoder
# =============================================================================
class SLICDecoder(nn.Module):
    """Transformer decoder with iterative box refinement (DINO-style).

    Args:
        decoder_layer: a single decoder layer (will be deep-copied)
        num_layers: number of decoder layers
        norm: optional final layer norm
        d_model: feature dimension
        return_intermediate: return outputs from all layers
        query_dim: dimension of reference points (4 for cx, cy, w, h)
    """

    def __init__(
        self,
        decoder_layer: nn.Module,
        num_layers: int,
        norm: Optional[nn.Module] = None,
        d_model: int = 256,
        return_intermediate: bool = True,
        query_dim: int = 4,
    ):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.norm = norm
        self.d_model = d_model
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        self.query_dim = query_dim

        # Conditional query position MLP (ref_point → query_pos)
        self.ref_point_head = MLP(query_dim // 2 * d_model, d_model, d_model, 2)

        # Will be set by DINO.__init__
        self.bbox_embed = None
        self.class_embed = None

    def forward(
        self,
        tgt: Tensor,                                    # [nq, bs, d_model]
        memory: Tensor,                                  # [N_mem, bs, d_model]
        memory_key_padding_mask: Optional[Tensor] = None, # [bs, N_mem]
        memory_pos: Optional[Tensor] = None,             # [N_mem, bs, d_model]
        refpoints_unsigmoid: Optional[Tensor] = None,    # [nq, bs, 4]
        tgt_mask: Optional[Tensor] = None,               # [nq, nq]
    ):
        """
        Returns:
            output: [num_layers, nq, bs, d_model] if return_intermediate
            ref_points: list of [nq, bs, 4] reference points per layer
        """
        output = tgt
        intermediate = []
        reference_points = refpoints_unsigmoid.sigmoid()
        ref_points = [reference_points]

        for layer_id, layer in enumerate(self.layers):
            # Generate query positional embedding from reference points
            query_sine_embed = gen_sineembed_for_position(reference_points)  # [nq, bs, 256*2]
            query_pos = self.ref_point_head(query_sine_embed)  # [nq, bs, d_model]

            output = layer(
                tgt=output,
                tgt_query_pos=query_pos,
                memory=memory,
                memory_key_padding_mask=memory_key_padding_mask,
                memory_pos=memory_pos,
                self_attn_mask=tgt_mask,
            )

            # Iterative box refinement
            if self.bbox_embed is not None:
                reference_before_sigmoid = inverse_sigmoid(reference_points)
                delta_unsig = self.bbox_embed[layer_id](output)
                outputs_unsig = delta_unsig + reference_before_sigmoid
                new_reference_points = outputs_unsig.sigmoid()
                reference_points = new_reference_points.detach()
                ref_points.append(new_reference_points)

            intermediate.append(self.norm(output))

        return [
            torch.stack(intermediate),   # [num_layers, nq, bs, d_model]
            torch.stack(ref_points),     # [num_layers+1, nq, bs, 4]
        ]


# =============================================================================
# SLIC Transformer (Top-Level)
# =============================================================================
class SLICTransformer(nn.Module):
    """Top-level transformer for superpixel tokens.

    Matches the interface expected by DINO.forward() as closely as possible.

    Args:
        d_model: feature dimension
        nhead: number of attention heads
        num_queries: number of object queries
        num_encoder_layers: number of encoder layers
        num_decoder_layers: number of decoder layers
        dim_feedforward: FFN hidden dimension
        dropout: dropout rate
        activation: activation function
        num_feature_levels: number of SLIC scale levels
        two_stage_type: 'no' or 'standard'
        return_intermediate_dec: return intermediate decoder outputs
        query_dim: reference point dimension (4)
        debug: debug flag
    """

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        num_queries: int = 900,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.0,
        activation: str = "relu",
        num_feature_levels: int = 3,
        two_stage_type: str = 'standard',
        two_stage_add_query_num: int = 0,
        two_stage_learn_wh: bool = False,
        return_intermediate_dec: bool = True,
        query_dim: int = 4,
        learnable_tgt_init: bool = True,
        embed_init_tgt: bool = True,
        random_refpoints_xy: bool = False,
        dec_pred_class_embed_share: bool = True,
        dec_pred_bbox_embed_share: bool = True,
        debug: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_queries = num_queries
        self.num_feature_levels = num_feature_levels
        self.two_stage_type = two_stage_type
        self.two_stage_add_query_num = two_stage_add_query_num
        self.two_stage_learn_wh = two_stage_learn_wh
        self.random_refpoints_xy = random_refpoints_xy
        self.embed_init_tgt = embed_init_tgt
        self.debug = debug

        # Positional encoding for superpixel centroids
        self.pos_embed = SuperpixelPositionEmbedding(num_pos_feats=d_model // 2)

        # Level embedding (to distinguish tokens from different SLIC scales)
        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))
        nn.init.normal_(self.level_embed)

        # Encoder
        encoder_layer = SLICEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation=activation, debug=debug,
        )
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder = SLICEncoder(
            encoder_layer, num_encoder_layers, norm=encoder_norm, d_model=d_model,
        )

        # Decoder
        decoder_layer = SLICDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation=activation, debug=debug,
        )
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = SLICDecoder(
            decoder_layer, num_decoder_layers, norm=decoder_norm,
            d_model=d_model, return_intermediate=return_intermediate_dec,
            query_dim=query_dim,
        )
        self.num_decoder_layers = num_decoder_layers

        # Two-stage (encoder output → proposals → decoder init)
        if two_stage_type == 'standard':
            self.enc_output = nn.Linear(d_model, d_model)
            self.enc_output_norm = nn.LayerNorm(d_model)
            if two_stage_learn_wh:
                self.two_stage_wh_embedding = nn.Embedding(1, 2)
            else:
                self.two_stage_wh_embedding = None
        if embed_init_tgt:
            self.tgt_embed = nn.Embedding(num_queries, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def gen_encoder_output_proposals_from_centroids(
        self,
        memory: Tensor,           # [bs, N, d_model]
        padding_mask: Tensor,     # [bs, N]
        centroids: Tensor,        # [bs, N, 2] — normalized (cx, cy)
        level_counts: List[int],  # per-level token counts
    ):
        """Generate box proposals from superpixel centroids.

        Each superpixel center → (cx, cy, w, h) where w/h are scale-dependent defaults.

        Returns:
            output_memory: [bs, N, d_model]
            output_proposals: [bs, N, 4] — unsigmoid
        """
        bs, N, C = memory.shape
        proposals = []
        offset = 0

        for lvl, count in enumerate(level_counts):
            level_centroids = centroids[:, offset:offset + count, :]  # [bs, count, 2]

            # Scale-dependent default width/height
            if self.two_stage_learn_wh and self.two_stage_wh_embedding is not None:
                wh = self.two_stage_wh_embedding.weight[0].sigmoid() * (2.0 ** lvl)
                wh = wh.unsqueeze(0).unsqueeze(0).expand(bs, count, -1)
            else:
                wh = torch.ones(bs, count, 2, device=memory.device) * 0.05 * (2.0 ** lvl)

            proposal = torch.cat([level_centroids, wh], dim=-1)  # [bs, count, 4]
            proposals.append(proposal)
            offset += count

        output_proposals = torch.cat(proposals, dim=1)  # [bs, N, 4]

        # Padding mask is the ONLY filter: padded positions must NEVER become
        # proposals. Real tokens — even those with edge centroids near 0 or 1 —
        # are legitimate superpixels and must be kept as valid proposals.
        # padding_mask: True = padded (invalid), False = real token.
        valid_mask = ~padding_mask.unsqueeze(-1)  # [bs, N, 1] — True = real/valid

        # Clamp proposals to safe range for inverse_sigmoid (prevents inf).
        # This does NOT discard edge superpixels — it just nudges their
        # coordinates slightly inward so log(x/(1-x)) stays finite.
        output_proposals = output_proposals.clamp(min=0.01, max=0.99)

        # Inverse sigmoid
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        # Mask out padded positions only
        output_proposals = output_proposals.masked_fill(~valid_mask, float('inf'))

        # Zero out memory for padded positions so they can't contribute
        output_memory = memory.masked_fill(~valid_mask, float(0))

        return output_memory, output_proposals

    def forward(
        self,
        backbone_output: dict,     # from GraphFeatureExtractor
        input_query_bbox=None,     # [bs, num_dn, 4] or None
        input_query_label=None,    # [bs, num_dn, d_model] or None
        attn_mask=None,            # [nq+num_dn, nq+num_dn] or None
    ):
        """
        Args:
            backbone_output: dict with tokens, centroids, padding_mask, level_counts
            input_query_bbox: DN reference points (from CDN)
            input_query_label: DN query embeddings (from CDN)
            attn_mask: attention mask for DN training

        Returns:
            hs:             [num_dec_layers, nq, bs, d_model]
            references:     [num_dec_layers+1, nq, bs, 4]
            hs_enc:         [1, bs, nq, d_model] or None
            ref_enc:        [1, bs, nq, 4] or None
            init_box_proposal: [bs, nq, 4] or None
        """
        tokens = backbone_output['tokens']           # [bs, N, d_model]
        centroids = backbone_output['centroids']     # [bs, N, 2]
        padding_mask = backbone_output['padding_mask']  # [bs, N]
        level_counts = backbone_output['level_counts']  # list[int]

        bs, N, d_model = tokens.shape

        # Compute positional encoding from centroids
        pos_embed = self.pos_embed(centroids)  # [bs, N, d_model]

        # Add level embeddings
        offset = 0
        for lvl, count in enumerate(level_counts):
            pos_embed[:, offset:offset + count] += self.level_embed[lvl].unsqueeze(0)
            offset += count

        # Transpose to [N, bs, d_model] for nn.MultiheadAttention
        src = tokens.permute(1, 0, 2)       # [N, bs, d_model]
        pos = pos_embed.permute(1, 0, 2)    # [N, bs, d_model]

        if self.debug:
            print(f"[SLICTransformer] Encoder input: src={src.shape}, pos={pos.shape}")

        # ===================== Encoder =====================
        memory = self.encoder(src, pos=pos, key_padding_mask=padding_mask)
        # memory: [N, bs, d_model]

        if self.debug:
            print(f"[SLICTransformer] Encoder output: memory={memory.shape}")

        # ===================== Two-Stage Proposals =====================
        hs_enc = ref_enc = init_box_proposal = None

        if self.two_stage_type == 'standard':
            memory_t = memory.permute(1, 0, 2)  # [bs, N, d_model]
            output_memory, output_proposals = self.gen_encoder_output_proposals_from_centroids(
                memory_t, padding_mask, centroids, level_counts,
            )
            output_memory = self.enc_output_norm(self.enc_output(output_memory))

            # Classification + regression on encoder output → select top-k
            enc_outputs_class = self.enc_out_class_embed(output_memory)
            enc_outputs_coord = self.enc_out_bbox_embed(output_memory) + output_proposals

            # Use padding_mask DIRECTLY to exclude padded positions from top-k.
            # padding_mask=True means padded → must never be selected as proposal.
            enc_outputs_class_for_topk = enc_outputs_class.clone()
            enc_outputs_class_for_topk.masked_fill_(padding_mask.unsqueeze(-1), float('-inf'))

            if self.debug:
                n_padded = padding_mask.sum(dim=1).float()
                print(f"[SLICTransformer] Proposals: total={output_proposals.shape[1]}, "
                      f"padded(mean)={n_padded.mean().item():.1f}, "
                      f"padded(max)={n_padded.max().item():.0f}")

            # Clamp topk to available valid (non-padded) tokens
            n_valid = (~padding_mask).sum(dim=1).min().item()
            topk = max(1, min(self.num_queries, int(n_valid)))
            topk_proposals = torch.topk(enc_outputs_class_for_topk.max(-1)[0], topk, dim=1)[1]

            # Gather reference points
            refpoint_embed_undetach = torch.gather(
                enc_outputs_coord, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4)
            )
            refpoint_embed_ = refpoint_embed_undetach.detach()
            init_box_proposal = torch.gather(
                output_proposals, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4)
            ).sigmoid()

            # Gather target embeddings
            tgt_undetach = torch.gather(
                output_memory, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, d_model)
            )

            if self.embed_init_tgt:
                tgt_ = self.tgt_embed.weight[:topk, None, :].repeat(1, bs, 1).transpose(0, 1)
            else:
                tgt_ = tgt_undetach.detach()

            if input_query_bbox is not None:
                refpoint_embed = torch.cat([input_query_bbox, refpoint_embed_], dim=1)
                tgt = torch.cat([input_query_label, tgt_], dim=1)
            else:
                refpoint_embed = refpoint_embed_
                tgt = tgt_

            # Encoder intermediate outputs (for auxiliary loss)
            hs_enc = tgt_undetach.unsqueeze(0)  # [1, bs, topk, d_model]
            ref_enc = refpoint_embed_undetach.sigmoid().clamp(min=1e-4, max=1 - 1e-4).unsqueeze(0)  # [1, bs, topk, 4]
        else:
            raise ValueError(f"two_stage_type={self.two_stage_type} not supported for SLIC")

        # ===================== Decoder =====================
        # [nq, bs, d_model]
        tgt = tgt.permute(1, 0, 2)
        # [nq, bs, 4]
        refpoint_embed = refpoint_embed.permute(1, 0, 2)
        # memory pos for cross-attention
        memory_pos = pos  # [N, bs, d_model]

        # Adjust attn_mask when topk < num_queries.
        # The mask from CDN is [dn_pad + num_queries, dn_pad + num_queries]
        # but actual decoder input is [dn_pad + topk] tokens.
        if attn_mask is not None and attn_mask.shape[0] != tgt.shape[0]:
            actual_nq = tgt.shape[0]  # dn_pad + topk
            dn_pad = attn_mask.shape[0] - self.num_queries
            # Keep all dn rows/cols, trim detection query portion to topk
            keep_rows = list(range(dn_pad)) + list(range(dn_pad, dn_pad + (actual_nq - dn_pad)))
            attn_mask = attn_mask[keep_rows][:, keep_rows]

        hs, references = self.decoder(
            tgt=tgt,
            memory=memory,
            memory_key_padding_mask=padding_mask,
            memory_pos=memory_pos,
            refpoints_unsigmoid=refpoint_embed,
            tgt_mask=attn_mask,
        )

        # Transpose from seq-first [n_layers, nq, bs, d] to batch-first [n_layers, bs, nq, d]
        # DINO expects batch-first throughout (matcher, criterion, losses)
        hs = hs.transpose(1, 2)
        references = references.transpose(1, 2)

        if self.debug:
            print(f"[SLICTransformer] Decoder output: hs={hs.shape}, refs={references.shape}")

        return hs, references, hs_enc, ref_enc, init_box_proposal
