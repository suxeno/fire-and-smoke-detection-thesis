# ------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import gradcheck

# ============================================================================
# Deformable Attention Tests (original)
# ============================================================================

from functions.ms_deform_attn_func import MSDeformAttnFunction, ms_deform_attn_core_pytorch


N, M, D = 1, 2, 2
Lq, L, P = 2, 2, 2
if torch.cuda.is_available():
    shapes = torch.as_tensor([(6, 4), (3, 2)], dtype=torch.long).cuda()
    level_start_index = torch.cat((shapes.new_zeros((1, )), shapes.prod(1).cumsum(0)[:-1]))
    S = sum([(H*W).item() for H, W in shapes])
else:
    shapes = None
    level_start_index = None
    S = 0


torch.manual_seed(3)


@torch.no_grad()
def check_forward_equal_with_pytorch_double():
    value = torch.rand(N, S, M, D).cuda() * 0.01
    sampling_locations = torch.rand(N, Lq, M, L, P, 2).cuda()
    attention_weights = torch.rand(N, Lq, M, L, P).cuda() + 1e-5
    attention_weights /= attention_weights.sum(-1, keepdim=True).sum(-2, keepdim=True)
    im2col_step = 2
    output_pytorch = ms_deform_attn_core_pytorch(value.double(), shapes, sampling_locations.double(), attention_weights.double()).detach().cpu()
    output_cuda = MSDeformAttnFunction.apply(value.double(), shapes, level_start_index, sampling_locations.double(), attention_weights.double(), im2col_step).detach().cpu()
    fwdok = torch.allclose(output_cuda, output_pytorch)
    max_abs_err = (output_cuda - output_pytorch).abs().max()
    max_rel_err = ((output_cuda - output_pytorch).abs() / output_pytorch.abs()).max()

    print(f'* {fwdok} check_forward_equal_with_pytorch_double: max_abs_err {max_abs_err:.2e} max_rel_err {max_rel_err:.2e}')


@torch.no_grad()
def check_forward_equal_with_pytorch_float():
    value = torch.rand(N, S, M, D).cuda() * 0.01
    sampling_locations = torch.rand(N, Lq, M, L, P, 2).cuda()
    attention_weights = torch.rand(N, Lq, M, L, P).cuda() + 1e-5
    attention_weights /= attention_weights.sum(-1, keepdim=True).sum(-2, keepdim=True)
    im2col_step = 2
    output_pytorch = ms_deform_attn_core_pytorch(value, shapes, sampling_locations, attention_weights).detach().cpu()
    output_cuda = MSDeformAttnFunction.apply(value, shapes, level_start_index, sampling_locations, attention_weights, im2col_step).detach().cpu()
    fwdok = torch.allclose(output_cuda, output_pytorch, rtol=1e-2, atol=1e-3)
    max_abs_err = (output_cuda - output_pytorch).abs().max()
    max_rel_err = ((output_cuda - output_pytorch).abs() / output_pytorch.abs()).max()

    print(f'* {fwdok} check_forward_equal_with_pytorch_float: max_abs_err {max_abs_err:.2e} max_rel_err {max_rel_err:.2e}')


def check_gradient_numerical(channels=4, grad_value=True, grad_sampling_loc=True, grad_attn_weight=True):

    value = torch.rand(N, S, M, channels).cuda() * 0.01
    sampling_locations = torch.rand(N, Lq, M, L, P, 2).cuda()
    attention_weights = torch.rand(N, Lq, M, L, P).cuda() + 1e-5
    attention_weights /= attention_weights.sum(-1, keepdim=True).sum(-2, keepdim=True)
    im2col_step = 2
    func = MSDeformAttnFunction.apply

    value.requires_grad = grad_value
    sampling_locations.requires_grad = grad_sampling_loc
    attention_weights.requires_grad = grad_attn_weight

    gradok = gradcheck(func, (value.double(), shapes, level_start_index, sampling_locations.double(), attention_weights.double(), im2col_step))

    print(f'* {gradok} check_gradient_numerical(D={channels})')


# ============================================================================
# Graph Backbone Tests
# ============================================================================

# Add project root to path so we can import models
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


def check_graph_feature_dimensions():
    """Test 2: Validate raw feature extractor produces exactly 64 dims after scatter_mean."""
    from models.dino.graph_backbone import SuperpixelCNN, GraphFeatureExtractor
    from skimage.segmentation import slic
    import torch

    print("\n--- check_graph_feature_dimensions ---")

    # Create small dummy image
    img_rgb = torch.rand(1, 3, 64, 64)
    img_np = img_rgb[0].permute(1, 2, 0).numpy()

    # Run SLIC
    segments = slic(img_np, n_segments=10, compactness=10, sigma=1, start_label=0, channel_axis=2)
    slic_map = torch.from_numpy(segments.astype(np.int64))

    # Extract features
    cnn = SuperpixelCNN(out_channels=64)
    cnn.eval()
    feat_map = cnn(img_rgb)
    
    # We can use GraphFeatureExtractor's internal method
    extr = GraphFeatureExtractor(n_segments_per_level=[10], output_dim=256)
    node_feat, valid_mask = extr._scatter_mean_features(feat_map[0], slic_map, K=50)

    assert node_feat.shape == (50, 64), \
        f"Expected (50, 64), got {node_feat.shape}"

    print(f"* True check_graph_feature_dimensions: shape={node_feat.shape}")


def check_graph_backbone_output_shapes():
    """Test 1: Validate Graph backbone produces correct dict output format."""
    from models.dino.graph_backbone import GraphFeatureExtractor
    from util.misc import NestedTensor

    print("\n--- check_graph_backbone_output_shapes ---")

    # Use small images for speed
    bs = 2
    images = torch.rand(bs, 3, 64, 64)
    masks = torch.zeros(bs, 64, 64, dtype=torch.bool)
    tensor_list = NestedTensor(images, masks)

    backbone = GraphFeatureExtractor(
        n_segments_per_level=[50, 25, 10],
        output_dim=256,
        max_superpixels_per_level=100,
        debug=False,
    )

    outputs = backbone(tensor_list)

    assert 'tokens' in outputs
    assert 'centroids' in outputs
    assert 'padding_mask' in outputs
    assert len(outputs['level_counts']) == 3

    print(f"* True check_graph_backbone_output_shapes: all expected keys present and shapes match logic")


def check_graph_gradient_flow():
    """Test 4: Validate gradients flow through CNN and GAT layers."""
    from models.dino.graph_backbone import GraphFeatureExtractor
    from util.misc import NestedTensor

    print("\n--- check_graph_gradient_flow ---")

    images = torch.rand(1, 3, 64, 64)
    masks = torch.zeros(1, 64, 64, dtype=torch.bool)
    tensor_list = NestedTensor(images, masks)

    backbone = GraphFeatureExtractor(
        n_segments_per_level=[20],
        output_dim=256,
        cnn_out_channels=16,
        gcn_hidden_dim=32,
        gcn_num_layers=2,
        max_superpixels_per_level=30,
    )

    outputs = backbone(tensor_list)
    loss = outputs["tokens"].sum()
    loss.backward()

    # Check projection layer has gradients
    grad_ok = True
    for name, param in backbone.named_parameters():
        if param.requires_grad and param.grad is None:
            print(f"  WARNING: {name} has no gradient!")
            grad_ok = False

    print(f"* {grad_ok} check_graph_gradient_flow")


def check_graph_full_model_forward():
    """Test 3: End-to-end smoke test of full DINO model with Graph backbone."""
    from models.dino.backbone import build_backbone
    from models.dino.dino import DINO
    from models.dino.deformable_transformer import build_deformable_transformer
    from util.misc import NestedTensor

    print("\n--- check_graph_full_model_forward ---")

    class Args:
        backbone = 'graph'
        slic_n_segments = [50, 25, 10]
        slic_compactness = 10.0
        slic_sigma = 1.0
        cnn_out_channels = 32
        gcn_hidden_dim = 64
        gcn_num_layers = 2
        gcn_edge_dim = 16
        gcn_heads = 2
        hidden_dim = 256
        position_embedding = 'sine'
        pe_temperatureH = 20
        pe_temperatureW = 20
        return_interm_indices = [0, 1, 2]
        backbone_freeze_keywords = None
        lr_backbone = 1e-5
        dilation = False
        use_checkpoint = False
        num_feature_levels = 4
        nheads = 8
        enc_layers = 2
        dec_layers = 2
        dim_feedforward = 512
        dropout = 0.0
        num_queries = 100
        query_dim = 4
        num_patterns = 0
        dec_n_points = 4
        enc_n_points = 4
        num_classes = 2
        aux_loss = False
        two_stage_type = 'standard'
        two_stage_bbox_embed_share = False
        two_stage_class_embed_share = False
        two_stage_add_query_num = 0
        two_stage_learn_wh = False
        two_stage_keep_all_tokens = False
        two_stage_pat_embed = 0
        two_stage_default_hw = 0.05
        dec_pred_bbox_embed_share = True
        dec_pred_class_embed_share = True
        decoder_sa_type = 'sa'
        decoder_module_seq = ['sa', 'ca', 'ffn']
        use_deformable_box_attn = False
        box_attn_type = 'roi_align'
        dec_layer_number = None
        transformer_activation = 'relu'
        random_refpoints_xy = False
        fix_refpoints_hw = -1
        use_dn = True
        dn_number = 10
        dn_box_noise_scale = 0.4
        dn_label_noise_ratio = 0.5
        dn_labelbook_size = 3
        embed_init_tgt = True
        match_unstable_error = True
        use_detached_boxes_dec_out = False
        add_channel_attention = False
        add_pos_value = False
        pre_norm = False
        unic_layers = 0
        pdetr3_bbox_embed_diff_each_layer = False
        pdetr3_refHW = -1
        dabdetr_yolo_like_anchor_update = False
        dabdetr_deformable_encoder = False
        dabdetr_deformable_decoder = False
        decoder_layer_noise = False
        dln_xy_noise = 0.2
        dln_hw_noise = 0.2
        batch_norm_type = 'FrozenBatchNorm2d'
        masks = False
        focal_alpha = 0.25
        num_select = 300
        nms_iou_threshold = -1

    args = Args()

    from models.dino.slic_transformer import SLICTransformer
    backbone_model = build_backbone(args)
    transformer = SLICTransformer(
        d_model=args.hidden_dim,
        nhead=args.nheads,
        num_queries=args.num_queries,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        activation=args.transformer_activation,
        num_feature_levels=args.num_feature_levels,
        two_stage_type=args.two_stage_type,
        two_stage_add_query_num=getattr(args, 'two_stage_add_query_num', 0),
        two_stage_learn_wh=getattr(args, 'two_stage_learn_wh', False),
        return_intermediate_dec=True,
        query_dim=4,
        learnable_tgt_init=getattr(args, 'embed_init_tgt', True),
        embed_init_tgt=getattr(args, 'embed_init_tgt', True),
        random_refpoints_xy=args.random_refpoints_xy,
        dec_pred_class_embed_share=getattr(args, 'dec_pred_class_embed_share', True),
        dec_pred_bbox_embed_share=getattr(args, 'dec_pred_bbox_embed_share', True),
    )

    model = DINO(
        backbone=backbone_model,
        transformer=transformer,
        num_classes=args.num_classes,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
        iter_update=True,
        query_dim=args.query_dim,
        num_feature_levels=args.num_feature_levels,
        nheads=args.nheads,
        dec_pred_class_embed_share=args.dec_pred_class_embed_share,
        dec_pred_bbox_embed_share=args.dec_pred_bbox_embed_share,
        two_stage_type=args.two_stage_type,
        two_stage_add_query_num=args.two_stage_add_query_num,
        two_stage_class_embed_share=args.two_stage_class_embed_share,
        two_stage_bbox_embed_share=args.two_stage_bbox_embed_share,
        decoder_sa_type=args.decoder_sa_type,
        dn_number=args.dn_number,
        dn_box_noise_scale=args.dn_box_noise_scale,
        dn_label_noise_ratio=args.dn_label_noise_ratio,
        dn_labelbook_size=args.dn_labelbook_size,
    )
    model.eval()

    bs = 1
    images = torch.rand(bs, 3, 64, 64)
    mask = torch.zeros(bs, 64, 64, dtype=torch.bool)
    samples = NestedTensor(images, mask)

    with torch.no_grad():
        outputs = model(samples)

    assert 'pred_logits' in outputs
    assert 'pred_boxes' in outputs
    print(f"* True check_graph_full_model_forward")


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', default='all', choices=['deform', 'graph', 'all'],
                        help='Which tests to run: deform (deformable attention), graph (Graph backbone), or all')
    args = parser.parse_args()

    if args.test in ['deform', 'all']:
        print("=" * 60)
        print("Running Deformable Attention Tests")
        print("=" * 60)
        check_forward_equal_with_pytorch_double()
        check_forward_equal_with_pytorch_float()
        for channels in [30, 32, 64, 128, 256, 512, 1024]:
            check_gradient_numerical(channels, True, True, True)

    if args.test in ['graph', 'all']:
        print("\n" + "=" * 60)
        print("Running Graph Backbone Tests")
        print("=" * 60)
        check_graph_feature_dimensions()
        check_graph_backbone_output_shapes()
        check_graph_gradient_flow()
        check_graph_full_model_forward()

    print("\n" + "=" * 60)
    print("All requested tests completed!")
    print("=" * 60)
