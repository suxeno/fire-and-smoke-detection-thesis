"""
Quick test for DINO-Superpixel-Graph (GAT backbone architecture).
Tests the full pipeline: CNN → scatter_mean → GAT → transformer → detection outputs.

Usage:
    cd code/DINO-Superpixel-Graph
    python quick_test_graph.py
"""

import sys
import os
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_backbone_output():
    """Test that GraphFeatureExtractor outputs token sequences + centroids."""
    from models.dino.graph_backbone import GraphFeatureExtractor
    from util.misc import NestedTensor

    print("=" * 60)
    print("TEST 1: Graph Backbone Token Output")
    print("=" * 60)

    backbone = GraphFeatureExtractor(
        n_segments_per_level=[20, 10, 5],
        output_dim=256,
        cnn_out_channels=32,
        gcn_hidden_dim=64,
        gcn_num_layers=2,
        gcn_edge_dim=16,
        gcn_heads=2,
        max_superpixels_per_level=50,
        debug=True,
    )

    # Fake image batch
    bs = 2
    images = torch.rand(bs, 3, 64, 64)
    mask = torch.zeros(bs, 64, 64, dtype=torch.bool)
    samples = NestedTensor(images, mask)

    # Create mock targets with multi-scale slic_maps
    targets = []
    for _ in range(bs):
        targets.append({
            'slic_maps': {
                20: torch.randint(0, 20, (64, 64)),
                10: torch.randint(0, 10, (64, 64)),
                5:  torch.randint(0, 5, (64, 64)),
            }
        })

    out = backbone(samples, targets=targets)

    assert isinstance(out, dict), f"Expected dict, got {type(out)}"
    assert 'tokens' in out, "Missing 'tokens' key"
    assert 'centroids' in out, "Missing 'centroids' key"
    assert 'padding_mask' in out, "Missing 'padding_mask' key"
    assert 'level_counts' in out, "Missing 'level_counts' key"

    tokens = out['tokens']
    centroids = out['centroids']
    padding_mask = out['padding_mask']
    level_counts = out['level_counts']

    print(f"  tokens:       {tokens.shape}")
    print(f"  centroids:    {centroids.shape}")
    print(f"  padding_mask: {padding_mask.shape}")
    print(f"  level_counts: {level_counts}")

    assert tokens.dim() == 3 and tokens.shape[0] == bs and tokens.shape[2] == 256
    assert centroids.dim() == 3 and centroids.shape[2] == 2
    assert tokens.shape[1] == centroids.shape[1] == padding_mask.shape[1]
    assert sum(level_counts) == tokens.shape[1]

    # Check centroids are in [0, 1]
    valid_centroids = centroids[~padding_mask]
    if valid_centroids.numel() > 0:
        assert valid_centroids.min() >= 0 and valid_centroids.max() <= 1, \
            f"Centroids out of range: [{valid_centroids.min():.4f}, {valid_centroids.max():.4f}]"

    print("  ✓ PASSED\n")
    return out


def test_transformer_forward():
    """Test SLICTransformer forward pass with fake backbone output."""
    from models.dino.slic_transformer import SLICTransformer

    print("=" * 60)
    print("TEST 2: SLICTransformer Forward (with graph backbone output)")
    print("=" * 60)

    bs = 2
    n_tokens = 35

    transformer = SLICTransformer(
        d_model=256,
        nhead=8,
        num_queries=10,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=512,
        num_feature_levels=3,
        two_stage_type='standard',
    )

    # Inject class/bbox heads (normally done by DINO.__init__)
    from models.dino.utils import MLP
    import copy, math

    _class_embed = torch.nn.Linear(256, 3)
    _bbox_embed = MLP(256, 256, 4, 3)
    prior_prob = 0.01
    bias_value = -math.log((1 - prior_prob) / prior_prob)
    _class_embed.bias.data = torch.ones(3) * bias_value

    transformer.enc_out_class_embed = _class_embed
    transformer.enc_out_bbox_embed = _bbox_embed
    transformer.decoder.bbox_embed = torch.nn.ModuleList(
        [copy.deepcopy(_bbox_embed) for _ in range(2)]
    )
    transformer.decoder.class_embed = torch.nn.ModuleList(
        [copy.deepcopy(_class_embed) for _ in range(2)]
    )

    # Fake backbone output (same format as GraphFeatureExtractor)
    backbone_output = {
        'tokens': torch.randn(bs, n_tokens, 256),
        'centroids': torch.rand(bs, n_tokens, 2),
        'padding_mask': torch.zeros(bs, n_tokens, dtype=torch.bool),
        'level_counts': [20, 10, 5],
    }

    hs, references, hs_enc, ref_enc, init_box_proposal = transformer(backbone_output)

    print(f"  hs:             {hs.shape}")
    print(f"  references:     {references.shape}")
    print(f"  hs_enc:         {hs_enc.shape if hs_enc is not None else None}")
    print(f"  ref_enc:        {ref_enc.shape if ref_enc is not None else None}")
    print(f"  init_box_prop:  {init_box_proposal.shape if init_box_proposal is not None else None}")

    assert hs.shape == (2, bs, 10, 256), f"Unexpected hs shape: {hs.shape}"
    assert references.shape[1] == bs and references.shape[2] == 10 and references.shape[3] == 4

    print("  ✓ PASSED\n")


def test_full_model():
    """Test end-to-end DINO model with graph backbone."""
    from util.slconfig import SLConfig

    print("=" * 60)
    print("TEST 3: Full DINO-Superpixel-Graph Model (End-to-End)")
    print("=" * 60)

    # Load config
    config_path = "config/DINO/DINO_4scale_graph.py"
    args = SLConfig.fromfile(config_path)

    # Override for speed
    args.slic_n_segments = [20, 10, 5]
    args.num_queries = 10
    args.enc_layers = 1
    args.dec_layers = 1
    args.device = 'cpu'
    args.cnn_out_channels = 32
    args.gcn_hidden_dim = 64
    args.gcn_num_layers = 2
    args.gcn_edge_dim = 16
    args.gcn_heads = 2

    from models.dino.dino import build_dino
    model, criterion, postprocessors = build_dino(args)
    model.eval()

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params:     {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")

    # Forward pass
    from util.misc import NestedTensor
    bs = 1
    images = torch.rand(bs, 3, 64, 64)
    mask = torch.zeros(bs, 64, 64, dtype=torch.bool)
    samples = NestedTensor(images, mask)

    targets = [{
        'labels': torch.tensor([0, 1], dtype=torch.long),
        'boxes': torch.tensor([[0.3, 0.3, 0.2, 0.2], [0.7, 0.7, 0.1, 0.1]]),
        'slic_maps': {
            20: torch.randint(0, 20, (64, 64)),
            10: torch.randint(0, 10, (64, 64)),
            5:  torch.randint(0, 5, (64, 64)),
        },
    }]

    with torch.no_grad():
        outputs = model(samples, targets)

    print(f"  pred_logits: {outputs['pred_logits'].shape}")
    print(f"  pred_boxes:  {outputs['pred_boxes'].shape}")

    assert 'pred_logits' in outputs
    assert 'pred_boxes' in outputs
    assert outputs['pred_logits'].shape[0] == bs
    assert outputs['pred_boxes'].shape[0] == bs
    assert outputs['pred_boxes'].shape[2] == 4

    print("  ✓ PASSED\n")


def test_gradient_flow():
    """Test that gradients flow through CNN + GAT layers."""
    from models.dino.graph_backbone import GraphFeatureExtractor
    from util.misc import NestedTensor

    print("=" * 60)
    print("TEST 4: Gradient Flow Through CNN + GAT")
    print("=" * 60)

    backbone = GraphFeatureExtractor(
        n_segments_per_level=[10],
        output_dim=256,
        cnn_out_channels=32,
        gcn_hidden_dim=64,
        gcn_num_layers=2,
        gcn_edge_dim=16,
        gcn_heads=2,
        max_superpixels_per_level=20,
    )

    images = torch.rand(1, 3, 32, 32)
    mask = torch.zeros(1, 32, 32, dtype=torch.bool)
    samples = NestedTensor(images, mask)

    targets = [{
        'slic_maps': {
            10: torch.randint(0, 10, (32, 32)),
        }
    }]

    out = backbone(samples, targets=targets)
    loss = out['tokens'].sum()
    loss.backward()

    # Check CNN gradients
    cnn_grad_ok = True
    for name, param in backbone.cnn.named_parameters():
        if param.requires_grad and param.grad is None:
            print(f"  WARNING: CNN {name} has no gradient!")
            cnn_grad_ok = False

    # Check GAT gradients
    gat_grad_ok = True
    for name, param in backbone.gat.named_parameters():
        if param.requires_grad and param.grad is None:
            print(f"  WARNING: GAT {name} has no gradient!")
            gat_grad_ok = False

    # Check GraphConstructor edge projection gradient
    edge_grad_ok = True
    for name, param in backbone.graph_constructor.named_parameters():
        if param.requires_grad and param.grad is None:
            print(f"  WARNING: GraphConstructor {name} has no gradient!")
            edge_grad_ok = False

    if cnn_grad_ok:
        print("  CNN gradients: ✓ OK")
    if gat_grad_ok:
        print("  GAT gradients: ✓ OK")
    if edge_grad_ok:
        print("  Edge projection gradients: ✓ OK")

    assert cnn_grad_ok, "CNN layers missing gradients!"
    assert gat_grad_ok, "GAT layers missing gradients!"

    print("  ✓ PASSED\n")


if __name__ == '__main__':
    print("\nDINO-Superpixel-Graph Quick Test (CNN + GAT backbone)\n")

    out = test_backbone_output()
    test_transformer_forward()
    test_full_model()
    test_gradient_flow()

    print("=" * 60)
    print("ALL TESTS PASSED ✓")
    print("=" * 60)
