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
shapes = torch.as_tensor([(6, 4), (3, 2)], dtype=torch.long).cuda()
level_start_index = torch.cat((shapes.new_zeros((1, )), shapes.prod(1).cumsum(0)[:-1]))
S = sum([(H*W).item() for H, W in shapes])


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
# SLIC Backbone Tests
# ============================================================================

# Add project root to path so we can import models
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


def check_slic_feature_dimensions():
    """Test 2: Validate raw feature extractor produces exactly 127 dims."""
    from models.dino.slic_backbone import SuperpixelFeatureComputer, RAW_FEATURE_DIM
    from skimage.segmentation import slic
    from skimage.color import rgb2hsv, rgb2lab

    print("\n--- check_slic_feature_dimensions ---")

    # Create small dummy image
    img_rgb = np.random.rand(64, 64, 3).astype(np.float32)
    img_hsv = rgb2hsv(img_rgb)
    img_lab = rgb2lab(img_rgb)
    img_gray = np.mean(img_rgb, axis=2)

    # Run SLIC
    segments = slic(img_rgb, n_segments=10, compactness=10, sigma=1, start_label=0, channel_axis=2)
    seg_id = np.unique(segments)[0]
    mask = segments == seg_id

    # Extract features
    features = SuperpixelFeatureComputer.extract_all_features(
        img_rgb, img_hsv, img_lab, img_gray,
        mask, segments, seg_id, 64, 64
    )

    assert features.shape == (RAW_FEATURE_DIM,), \
        f"Expected ({RAW_FEATURE_DIM},), got {features.shape}"
    assert not np.any(np.isnan(features)), "Features contain NaN!"
    assert not np.any(np.isinf(features)), "Features contain Inf!"

    print(f"* True check_slic_feature_dimensions: shape={features.shape}, "
          f"mean={features.mean():.4f}, std={features.std():.4f}")


def check_slic_backbone_output_shapes():
    """Test 1: Validate SLIC backbone produces correct multi-scale output shapes."""
    from models.dino.slic_backbone import SLICFeatureExtractor
    from util.misc import NestedTensor

    print("\n--- check_slic_backbone_output_shapes ---")

    # Use small images for speed
    bs = 2
    images = torch.rand(bs, 3, 128, 128)
    masks = torch.zeros(bs, 128, 128, dtype=torch.bool)
    tensor_list = NestedTensor(images, masks)

    grid_sizes = [(16, 16), (8, 8), (4, 4)]
    backbone = SLICFeatureExtractor(
        n_segments_per_level=[50, 25, 10],
        compactness=10,
        sigma=1,
        output_dim=256,
        grid_sizes=grid_sizes,
        debug=True,
    )

    outputs = backbone(tensor_list)

    assert len(outputs) == 3, f"Expected 3 levels, got {len(outputs)}"
    for level_idx, (grid_h, grid_w) in enumerate(grid_sizes):
        key = str(level_idx)
        assert key in outputs, f"Missing level {key}"
        nt = outputs[key]
        assert nt.tensors.shape == (bs, 256, grid_h, grid_w), \
            f"Level {key}: expected ({bs}, 256, {grid_h}, {grid_w}), got {nt.tensors.shape}"
        assert nt.mask.shape == (bs, grid_h, grid_w), \
            f"Level {key} mask: expected ({bs}, {grid_h}, {grid_w}), got {nt.mask.shape}"

    print(f"* True check_slic_backbone_output_shapes: all 3 levels correct")


def check_slic_projection_gradient():
    """Test 4: Validate gradients flow through the projection MLP."""
    from models.dino.slic_backbone import SLICFeatureExtractor
    from util.misc import NestedTensor

    print("\n--- check_slic_projection_gradient ---")

    images = torch.rand(1, 3, 64, 64)
    masks = torch.zeros(1, 64, 64, dtype=torch.bool)
    tensor_list = NestedTensor(images, masks)

    backbone = SLICFeatureExtractor(
        n_segments_per_level=[20],
        compactness=10,
        sigma=1,
        output_dim=256,
        grid_sizes=[(8, 8)],
    )

    outputs = backbone(tensor_list)
    loss = outputs["0"].tensors.sum()
    loss.backward()

    # Check projection layer has gradients
    grad_ok = True
    for name, param in backbone.projection.named_parameters():
        if param.grad is None:
            print(f"  WARNING: {name} has no gradient!")
            grad_ok = False
        else:
            print(f"  {name}: grad_mean={param.grad.mean():.6f}, grad_std={param.grad.std():.6f}")

    print(f"* {grad_ok} check_slic_projection_gradient")


def check_slic_full_model_forward():
    """Test 3: End-to-end smoke test of full DINO model with SLIC backbone."""
    from models.dino.backbone import build_backbone, Joiner
    from models.dino.dino import DINO
    from models.dino.deformable_transformer import build_deformable_transformer
    from util.misc import NestedTensor

    print("\n--- check_slic_full_model_forward ---")

    # Build a minimal args namespace mimicking DINO_4scale_slic config
    class Args:
        backbone = 'slic'
        slic_n_segments = [50, 25, 10]
        slic_compactness = 10.0
        slic_sigma = 1.0
        slic_grid_sizes = [(16, 16), (8, 8), (4, 4)]
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
        enc_layers = 2  # Small for testing
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

    # Build backbone
    backbone_model = build_backbone(args)
    print(f"  Backbone built, num_channels={backbone_model.num_channels}")

    # Build transformer
    transformer = build_deformable_transformer(args)
    print(f"  Transformer built, d_model={transformer.d_model}")

    # Build DINO
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
    print(f"  DINO model built successfully")

    # Create dummy input
    bs = 1
    images = torch.rand(bs, 3, 128, 128)
    mask = torch.zeros(bs, 128, 128, dtype=torch.bool)
    samples = NestedTensor(images, mask)

    # Forward pass (no targets in eval mode)
    with torch.no_grad():
        outputs = model(samples)

    assert 'pred_logits' in outputs, "Missing pred_logits in output"
    assert 'pred_boxes' in outputs, "Missing pred_boxes in output"
    print(f"  pred_logits: {outputs['pred_logits'].shape}")
    print(f"  pred_boxes: {outputs['pred_boxes'].shape}")
    assert outputs['pred_logits'].shape[-1] == args.num_classes
    assert outputs['pred_boxes'].shape[-1] == 4

    print(f"* True check_slic_full_model_forward")


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', default='all', choices=['deform', 'slic', 'all'],
                        help='Which tests to run: deform (deformable attention), slic (SLIC backbone), or all')
    args = parser.parse_args()

    if args.test in ['deform', 'all']:
        print("=" * 60)
        print("Running Deformable Attention Tests")
        print("=" * 60)
        check_forward_equal_with_pytorch_double()
        check_forward_equal_with_pytorch_float()
        for channels in [30, 32, 64, 128, 256, 512, 1024]:
            check_gradient_numerical(channels, True, True, True)

    if args.test in ['slic', 'all']:
        print("\n" + "=" * 60)
        print("Running SLIC Backbone Tests")
        print("=" * 60)
        check_slic_feature_dimensions()
        check_slic_backbone_output_shapes()
        check_slic_projection_gradient()
        check_slic_full_model_forward()

    print("\n" + "=" * 60)
    print("All requested tests completed!")
    print("=" * 60)
