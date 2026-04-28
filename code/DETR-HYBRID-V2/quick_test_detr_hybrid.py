import argparse
import torch
import torch.nn.functional as F
from models.detr import build

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', default='resnet50')
    parser.add_argument('--dilation', action='store_true')
    parser.add_argument('--position_embedding', default='sine')
    parser.add_argument('--enc_layers', default=6, type=int)
    parser.add_argument('--dec_layers', default=6, type=int)
    parser.add_argument('--dim_feedforward', default=2048, type=int)
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--nheads', default=8, type=int)
    parser.add_argument('--num_queries', default=100, type=int)
    parser.add_argument('--pre_norm', action='store_true')
    parser.add_argument('--masks', action='store_true')
    parser.add_argument('--aux_loss', action='store_true')
    parser.add_argument('--slic_n_segments', default=200, type=int)
    parser.add_argument('--pooling_type', default='mean', type=str)
    parser.add_argument('--hybrid_token_mode', default='mixed', type=str)
    parser.add_argument('--compact_superpixel_ids', action='store_true')
    parser.add_argument('--require_superpixels', action='store_true')

    # Efficiency-first pixel-token pruning
    parser.add_argument('--pixel_prune', action='store_true')
    parser.add_argument('--pixel_prune_keep_ratio', default=0.8, type=float)
    parser.add_argument('--pixel_prune_score_mode', default='saliency', type=str,
                        choices=('saliency', 'feature_norm', 'counts'))
    parser.add_argument('--pixel_prune_w_feature', default=0.45, type=float)
    parser.add_argument('--pixel_prune_w_color', default=0.25, type=float)
    parser.add_argument('--pixel_prune_w_texture', default=0.20, type=float)
    parser.add_argument('--pixel_prune_w_size', default=0.10, type=float)
    
    # criterion args
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--set_cost_class', default=1)
    parser.add_argument('--set_cost_bbox', default=5)
    parser.add_argument('--set_cost_giou', default=2)
    parser.add_argument('--bbox_loss_coef', default=5)
    parser.add_argument('--giou_loss_coef', default=2)
    parser.add_argument('--eos_coef', default=0.1)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--lr_backbone', default=1e-5, type=float) # for build
    
    args = parser.parse_args()
    
    print("Building model...")
    model, criterion, postprocs = build(args)
    
    print("Loading pretrained DETR-R50 weights...")
    checkpoint = torch.hub.load_state_dict_from_url("https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth", map_location='cpu')
    state_dict = checkpoint["model"]
    
    # Filter mismatch keys
    model_state = model.state_dict()
    for k in list(state_dict.keys()):
        if k in model_state and state_dict[k].shape != model_state[k].shape:
            print(f"Shapes do not match for {k}: {state_dict[k].shape} vs {model_state[k].shape}. Deleting key.")
            del state_dict[k]
    
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"Missing keys: {missing}")
    print(f"Unexpected keys: {unexpected}")
    
    # Dummy inputs
    bs = 2
    h, w = 800, 800
    images = torch.randn(bs, 3, h, w)
    mask = torch.zeros(bs, h, w, dtype=torch.bool)
    from util.misc import NestedTensor
    samples = NestedTensor(images, mask)
    
    # Dummy Targets with superpixels
    # Generate a dummy superpixel map [h, w] filled with clusters 0-199
    # To mimic it being downsampled efficiently, we create a smaller map and interpolate up
    sp_h, sp_w = h // 16, w // 16
    sp_vals = torch.randint(0, 200, (bs, sp_h, sp_w))
    sp_maps_full = F.interpolate(sp_vals.unsqueeze(1).float(), size=(h, w), mode='nearest').long().squeeze(1)
    
    targets = []
    for i in range(bs):
        t = {}
        t['labels'] = torch.tensor([1])
        t['boxes'] = torch.tensor([[0.5, 0.5, 0.2, 0.2]])
        t['slic_maps'] = {200: sp_maps_full[i]}
        targets.append(t)
    
    print("Forward Pass...")
    out = model(samples, targets, debug=True)
    
    print("Output shapes:")
    print("pred_logits:", out['pred_logits'].shape)
    print("pred_boxes:", out['pred_boxes'].shape)
    
    print("Loss Calculation...")
    loss_dict = criterion(out, targets)
    losses = sum(loss_dict.values())
    print("Total Loss:", losses.item())
    
    print("Done! Sanity test passed.")

if __name__ == "__main__":
    main()
