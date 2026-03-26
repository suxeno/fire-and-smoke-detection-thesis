#!/usr/bin/env python3
"""Test script to validate DINO-SLIC NaN fixes and diagnose remaining issues.

Tests:
  1. Padding mask fix: verifies padded positions are excluded from proposals
  2. Forward pass NaN scan: hooks every module to detect NaN at first occurrence
  3. Stress test: synthetic edge-case inputs to provoke NaN

Usage:
    # Quick validation of padding fix (no GPU needed, ~2s):
    python test_nan_fix.py --test padding

    # Full forward pass with real data and NaN hooks (needs GPU + dataset):
    python test_nan_fix.py --test forward

    # Stress test with synthetic extreme inputs (GPU, ~10s):
    python test_nan_fix.py --test stress

    # Run the EXACT batch that crashes training (GPU + dataset, ~30s):
    python test_nan_fix.py --test replay --replay-iter 10975

    # Run all tests:
    python test_nan_fix.py --test all
"""

import sys
import os
import argparse
import math
import torch
import torch.nn as nn

# Add project root so imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ─────────────────────────────────────────────────────────────────────────────
# TEST 1: Padding Mask Fix Validation
# ─────────────────────────────────────────────────────────────────────────────
def test_padding_fix():
    """Validate that padded positions are excluded from proposals and top-k."""
    from util.misc import inverse_sigmoid

    print("=" * 70)
    print("TEST 1: Padding Mask Fix Validation")
    print("=" * 70)

    bs, N, d_model = 2, 100, 256
    num_queries = 10
    device = "cpu"

    # Simulate: 60 valid tokens + 40 padded tokens
    n_valid = 60
    padding_mask = torch.zeros(bs, N, dtype=torch.bool, device=device)
    padding_mask[:, n_valid:] = True  # last 40 are padded

    centroids = torch.rand(bs, N, 2, device=device)
    centroids[:, n_valid:] = 0.0  # padded centroids at [0,0]

    memory = torch.randn(bs, N, d_model, device=device)

    # ── Reproduce gen_encoder_output_proposals_from_centroids logic ──
    level_counts = [N]  # single level for simplicity
    proposals = []
    offset = 0
    for lvl, count in enumerate(level_counts):
        level_centroids = centroids[:, offset:offset + count, :]
        wh = torch.ones(bs, count, 2, device=device) * 0.05 * (2.0 ** lvl)
        proposal = torch.cat([level_centroids, wh], dim=-1)
        proposals.append(proposal)
        offset += count
    output_proposals = torch.cat(proposals, dim=1)

    valid_mask = ~padding_mask.unsqueeze(-1)
    output_proposals = output_proposals.clamp(min=0.01, max=0.99)
    output_proposals = torch.log(output_proposals / (1 - output_proposals))
    output_proposals = output_proposals.masked_fill(~valid_mask, float('inf'))
    output_memory = memory.masked_fill(~valid_mask, 0.0)

    # ── Checks ──
    passed = True

    # Check 1a: Padded proposals should be inf
    padded_proposals = output_proposals[:, n_valid:]
    if not torch.isinf(padded_proposals).all():
        print("  FAIL: Padded proposals are NOT all inf")
        passed = False
    else:
        print("  PASS: Padded proposals are all inf")

    # Check 1b: Valid proposals should be finite
    valid_proposals = output_proposals[:, :n_valid]
    if not torch.isfinite(valid_proposals).all():
        print("  FAIL: Valid proposals contain inf/NaN")
        passed = False
    else:
        print("  PASS: Valid proposals are all finite")

    # Check 1c: Padded memory should be zero
    padded_memory = output_memory[:, n_valid:]
    if not (padded_memory == 0).all():
        print("  FAIL: Padded memory is NOT zeroed")
        passed = False
    else:
        print("  PASS: Padded memory is zeroed")

    # ── Simulate top-k selection ──
    enc_outputs_class = torch.randn(bs, N, 2, device=device)  # 2 classes
    enc_outputs_coord = torch.randn(bs, N, 4, device=device) + output_proposals

    enc_outputs_class_for_topk = enc_outputs_class.clone()
    enc_outputs_class_for_topk.masked_fill_(padding_mask.unsqueeze(-1), float('-inf'))

    n_valid_min = (~padding_mask).sum(dim=1).min().item()
    topk = max(1, min(num_queries, int(n_valid_min)))
    topk_indices = torch.topk(enc_outputs_class_for_topk.max(-1)[0], topk, dim=1)[1]

    # Check 1d: Top-k should only contain valid indices
    max_topk_idx = topk_indices.max().item()
    if max_topk_idx >= n_valid:
        print(f"  FAIL: Top-k selected padded index {max_topk_idx} (valid range 0-{n_valid-1})")
        passed = False
    else:
        print(f"  PASS: Top-k indices all < {n_valid} (max={max_topk_idx})")

    # Check 1e: Gathered proposals should be finite
    gathered_coord = torch.gather(enc_outputs_coord, 1, topk_indices.unsqueeze(-1).repeat(1, 1, 4))
    if not torch.isfinite(gathered_coord).all():
        print("  FAIL: Gathered coordinates contain inf/NaN")
        passed = False
    else:
        print("  PASS: Gathered coordinates are all finite")

    # Check 1f: Gathered proposals as sigmoid should be in (0,1)
    gathered_sigmoid = torch.gather(output_proposals, 1, topk_indices.unsqueeze(-1).repeat(1, 1, 4)).sigmoid()
    if gathered_sigmoid.min() < 0 or gathered_sigmoid.max() > 1:
        print(f"  FAIL: Sigmoid proposals out of range [{gathered_sigmoid.min():.4f}, {gathered_sigmoid.max():.4f}]")
        passed = False
    else:
        print(f"  PASS: Sigmoid proposals in valid range [{gathered_sigmoid.min():.4f}, {gathered_sigmoid.max():.4f}]")

    print(f"\n  {'ALL PASSED' if passed else 'SOME TESTS FAILED'}")
    return passed


# ─────────────────────────────────────────────────────────────────────────────
# TEST 2: Forward Pass NaN Detection (hooks on every module)
# ─────────────────────────────────────────────────────────────────────────────
class NaNDetector:
    """Register forward hooks on all modules to detect NaN/inf."""

    def __init__(self, model):
        self.nan_found = []
        self.hooks = []
        for name, module in model.named_modules():
            h = module.register_forward_hook(self._make_hook(name))
            self.hooks.append(h)

    def _make_hook(self, name):
        def hook(module, inp, out):
            def check(tensor, tag):
                if isinstance(tensor, torch.Tensor):
                    if torch.isnan(tensor).any():
                        self.nan_found.append((name, tag, "NaN",
                                               tensor.shape,
                                               torch.isnan(tensor).sum().item()))
                    if torch.isinf(tensor).any():
                        self.nan_found.append((name, tag, "Inf",
                                               tensor.shape,
                                               torch.isinf(tensor).sum().item()))

            if isinstance(out, torch.Tensor):
                check(out, "output")
            elif isinstance(out, (tuple, list)):
                for i, o in enumerate(out):
                    if isinstance(o, torch.Tensor):
                        check(o, f"output[{i}]")
        return hook

    def remove(self):
        for h in self.hooks:
            h.remove()

    def report(self):
        if not self.nan_found:
            print("  No NaN/Inf detected in any module output")
            return True
        print(f"  Found {len(self.nan_found)} NaN/Inf occurrences:")
        for name, tag, kind, shape, count in self.nan_found:
            print(f"    {kind} in {name}.{tag} shape={shape} count={count}")
        return False


def build_model_from_config():
    """Build the DINO-SLIC model using the same config as training."""
    from util.slconfig import SLConfig
    from models.dino import build_dino

    config_path = os.path.join(os.path.dirname(__file__),
                               "config/DINO/DINO_4scale_slic.py")
    cfg = SLConfig.fromfile(config_path)

    # Minimal args needed by build_dino
    args = cfg.copy()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.masks = False
    args.aux_loss = True

    # Dataset args (needed by build_dataset / replay test)
    args.dataset_file = 'coco'
    args.coco_path = '/home/Media/Dataset/FASDD/FASDD_CV'
    args.fix_size = False

    model, criterion, postprocessors = build_dino(args)
    model = model.to(args.device)
    return model, criterion, args


def build_dataloader(args):
    """Build the same data loader as training."""
    from datasets import build_dataset
    import util.misc as utils

    dataset_train = build_dataset(image_set='train', args=args)
    sampler = torch.utils.data.SequentialSampler(dataset_train)
    loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        sampler=sampler,
        drop_last=True,
        collate_fn=utils.collate_fn,
        num_workers=0,  # single-process for reproducibility
    )
    return loader


def test_forward_nan():
    """Build model + load one real batch, run forward with NaN detection hooks."""
    print("=" * 70)
    print("TEST 2: Forward Pass NaN Detection")
    print("=" * 70)

    model, criterion, args = build_model_from_config()
    model.train()

    detector = NaNDetector(model)

    # Create a synthetic batch that mimics real data
    device = args.device
    bs = 2
    C, H, W = 3, 640, 640
    samples = torch.randn(bs, C, H, W, device=device)

    # Normalize like ImageNet
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    samples = (samples * std + mean).clamp(0, 1)
    samples = (samples - mean) / std

    # Create targets
    targets = []
    for _ in range(bs):
        n_boxes = torch.randint(1, 5, (1,)).item()
        targets.append({
            'labels': torch.randint(0, args.num_classes, (n_boxes,), device=device),
            'boxes': torch.rand(n_boxes, 4, device=device).clamp(0.05, 0.95),
        })
        # Ensure valid cxcywh (w,h > 0)
        targets[-1]['boxes'][:, 2:] = targets[-1]['boxes'][:, 2:].clamp(min=0.05)

    print("  Running forward pass with NaN hooks...")
    try:
        with torch.no_grad():
            outputs = model(samples, targets)
        print("  Forward pass completed without crash")

        # Check outputs
        pred_boxes = outputs['pred_boxes']
        has_nan = torch.isnan(pred_boxes).any().item()
        has_inf = torch.isinf(pred_boxes).any().item()
        print(f"  pred_boxes: shape={pred_boxes.shape}, "
              f"nan={has_nan}, inf={has_inf}, "
              f"min={pred_boxes[~torch.isnan(pred_boxes)].min():.4f}, "
              f"max={pred_boxes[~torch.isnan(pred_boxes)].max():.4f}")
    except Exception as e:
        print(f"  Forward pass CRASHED: {e}")

    ok = detector.report()
    detector.remove()
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# TEST 3: Stress Test with Edge Cases
# ─────────────────────────────────────────────────────────────────────────────
def test_stress():
    """Stress test the model with pathological inputs."""
    print("=" * 70)
    print("TEST 3: Stress Test with Edge Cases")
    print("=" * 70)

    model, criterion, args = build_model_from_config()
    model.train()

    device = args.device

    test_cases = [
        ("Normal image", False),
        ("Near-black image", False),
        ("Near-white image", False),
        ("High-contrast stripes", False),
        ("Tiny image (resized)", False),
    ]

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    all_passed = True
    for case_name, _ in test_cases:
        bs = 2
        H, W = 640, 640

        if case_name == "Normal image":
            raw = torch.rand(bs, 3, H, W, device=device)
        elif case_name == "Near-black image":
            raw = torch.full((bs, 3, H, W), 0.01, device=device)
            raw += torch.randn_like(raw) * 0.005
            raw.clamp_(0, 1)
        elif case_name == "Near-white image":
            raw = torch.full((bs, 3, H, W), 0.99, device=device)
            raw += torch.randn_like(raw) * 0.005
            raw.clamp_(0, 1)
        elif case_name == "High-contrast stripes":
            raw = torch.zeros(bs, 3, H, W, device=device)
            raw[:, :, ::2, :] = 1.0  # alternating rows
        elif case_name == "Tiny image (resized)":
            tiny = torch.rand(bs, 3, 32, 32, device=device)
            raw = torch.nn.functional.interpolate(tiny, size=(H, W), mode='bilinear')

        samples = (raw - mean) / std  # normalize

        targets = [{
            'labels': torch.tensor([0], device=device),
            'boxes': torch.tensor([[0.5, 0.5, 0.3, 0.3]], device=device),
        } for _ in range(bs)]

        print(f"  [{case_name}]...", end=" ")
        try:
            with torch.no_grad():
                outputs = model(samples, targets)
            pb = outputs['pred_boxes']
            has_nan = torch.isnan(pb).any().item()
            has_inf = torch.isinf(pb).any().item()
            if has_nan or has_inf:
                print(f"FAIL (nan={has_nan}, inf={has_inf})")
                all_passed = False
            else:
                print(f"PASS (range=[{pb.min():.4f}, {pb.max():.4f}])")
        except Exception as e:
            print(f"CRASH: {e}")
            all_passed = False

    print(f"\n  {'ALL CASES PASSED' if all_passed else 'SOME CASES FAILED'}")
    return all_passed


# ─────────────────────────────────────────────────────────────────────────────
# TEST 4: Replay Exact Training Batch
# ─────────────────────────────────────────────────────────────────────────────
def test_replay(target_iter=10975):
    """Replay the exact batch that crashes during training."""
    print("=" * 70)
    print(f"TEST 4: Replay Training Batch (iteration {target_iter})")
    print("=" * 70)

    model, criterion, args = build_model_from_config()
    model.train()

    # Set same seed as training
    torch.manual_seed(42)

    device = args.device
    loader = build_dataloader(args)

    # We can't replicate exact model state at iter N without checkpoints,
    # but we CAN find the exact data batch and test it.
    print(f"  Scanning to iteration {target_iter}...")

    detector = NaNDetector(model)

    for i, (samples, targets) in enumerate(loader):
        if i < target_iter - 5:
            # Fast-forward, but still do a forward pass for the last few
            # to accumulate some gradient state
            continue

        samples = samples.to(device)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in t.items()} for t in targets]

        if i >= target_iter - 5:
            print(f"  Iter {i}: running forward...", end=" ")
            detector.nan_found.clear()

            try:
                outputs = model(samples, targets)
                loss_dict = criterion(outputs, targets)
                losses = sum(loss_dict[k] * v for k, v in criterion.weight_dict.items()
                             if k in loss_dict)
                loss_val = losses.item()

                pb = outputs['pred_boxes']
                has_nan = torch.isnan(pb).any().item()
                print(f"loss={loss_val:.4f}, pred_boxes nan={has_nan}")

                if has_nan:
                    detector.report()
                    # Detailed diagnostics on the batch
                    print(f"\n  Batch diagnostics:")
                    for ti, t in enumerate(targets):
                        print(f"    Image {ti}: {len(t['labels'])} boxes, "
                              f"labels={t['labels'].tolist()}")
                    break

                # Do a backward + optimizer step to advance model state
                losses.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                # Use a dummy optimizer step
                for p in model.parameters():
                    if p.grad is not None:
                        p.data -= 1e-4 * p.grad
                        p.grad = None

            except Exception as e:
                print(f"CRASH: {e}")
                detector.report()
                break

        if i >= target_iter:
            break

    detector.remove()


# ─────────────────────────────────────────────────────────────────────────────
# TEST 5: Multi-iteration NaN accumulation test
# ─────────────────────────────────────────────────────────────────────────────
def test_training_loop(n_iters=200):
    """Run a short training loop with NaN detection at every step.

    Uses the real dataset and collate_fn (NestedTensor) to match actual
    training conditions, including images with zero GT boxes.
    """
    print("=" * 70)
    print(f"TEST 5: Training Loop NaN Detection ({n_iters} iterations)")
    print("=" * 70)

    model, criterion, args = build_model_from_config()
    model.train()
    device = args.device

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)

    loader = build_dataloader(args)
    data_iter = iter(loader)

    nan_iter = None
    for i in range(n_iters):
        # Get a real batch (wraps around if dataset is exhausted)
        try:
            samples, targets = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            samples, targets = next(data_iter)

        samples = samples.to(device)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in t.items()} for t in targets]

        try:
            outputs = model(samples, targets)
            loss_dict = criterion(outputs, targets)
            losses = sum(loss_dict[k] * criterion.weight_dict[k]
                         for k in loss_dict if k in criterion.weight_dict)

            # Check for NaN in loss
            if not math.isfinite(losses.item()):
                print(f"  Iter {i}: Loss is {losses.item()} — STOPPING")
                nan_iter = i
                break

            # Check for NaN in pred_boxes
            pb = outputs['pred_boxes']
            if torch.isnan(pb).any() or torch.isinf(pb).any():
                print(f"  Iter {i}: NaN/Inf in pred_boxes — STOPPING")
                nan_iter = i
                break

            optimizer.zero_grad()
            losses.backward()

            # Check gradients for NaN
            grad_nan = False
            for name, p in model.named_parameters():
                if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                    print(f"  Iter {i}: NaN/Inf gradient in {name}")
                    grad_nan = True
                    break

            if grad_nan:
                print(f"  Iter {i}: Gradient NaN detected — STOPPING")
                nan_iter = i
                break

            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()

            if i % 50 == 0:
                print(f"  Iter {i}: loss={losses.item():.4f}")

        except Exception as e:
            print(f"  Iter {i}: CRASH: {e}")
            import traceback
            traceback.print_exc()
            nan_iter = i
            break

    if nan_iter is None:
        print(f"\n  PASSED: No NaN detected in {n_iters} iterations")
        return True
    else:
        print(f"\n  FAILED: NaN detected at iteration {nan_iter}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DINO-SLIC NaN Fix Tests")
    parser.add_argument("--test", type=str, default="padding",
                        choices=["padding", "forward", "stress", "replay", "train", "all"],
                        help="Which test to run")
    parser.add_argument("--replay-iter", type=int, default=10975,
                        help="Target iteration for replay test")
    parser.add_argument("--train-iters", type=int, default=200,
                        help="Number of iterations for training loop test")
    args = parser.parse_args()

    results = {}

    if args.test in ("padding", "all"):
        results["padding"] = test_padding_fix()

    if args.test in ("forward", "all"):
        results["forward"] = test_forward_nan()

    if args.test in ("stress", "all"):
        results["stress"] = test_stress()

    if args.test in ("replay",):
        test_replay(args.replay_iter)

    if args.test in ("train", "all"):
        results["train"] = test_training_loop(args.train_iters)

    if results:
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        for name, ok in results.items():
            print(f"  {name}: {'PASS' if ok else 'FAIL'}")
