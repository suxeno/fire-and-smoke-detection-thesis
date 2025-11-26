# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Transforms and data augmentation for both image + bbox.
Simplified for fire/smoke detection - minimal augmentation, focus on normalization and resizing.
"""
import random

import PIL
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F

from .box_ops import box_xyxy_to_cxcywh
from .misc import interpolate

def resize(image, target, size, max_size=None):
    """Resize image and adjust bounding boxes accordingly."""
    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)

    size = get_size(image.size, size, max_size)
    rescaled_image = F.resize(image, size)

    if target is None:
        return rescaled_image, None

    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_image.size, image.size))
    ratio_width, ratio_height = ratios

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        scaled_boxes = boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
        target["boxes"] = scaled_boxes

    if "area" in target:
        area = target["area"]
        scaled_area = area * (ratio_width * ratio_height)
        target["area"] = scaled_area

    h, w = size
    target["size"] = torch.tensor([h, w])

    if "masks" in target:
        target['masks'] = interpolate(
            target['masks'][:, None].float(), size, mode="nearest")[:, 0] > 0.5

    if "slic_map" in target:
        # slic_map is (H, W), need (1, 1, H, W) for interpolate
        # Use nearest neighbor to preserve integer IDs
        slic_map = target["slic_map"].unsqueeze(0).unsqueeze(0).float()
        slic_map = interpolate(slic_map, size, mode="nearest")
        target["slic_map"] = slic_map.squeeze().long()

    return rescaled_image, target


class RandomResize(object):
    """Randomly resize to one of the given sizes (for multi-scale training)."""
    def __init__(self, sizes, max_size=None):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, img, target=None):
        size = random.choice(self.sizes)
        return resize(img, target, size, self.max_size)


class ToTensor(object):
    """Convert PIL Image to Tensor."""
    def __call__(self, img, target):
        return F.to_tensor(img), target


class Normalize(object):
    """Normalize image and convert boxes to normalized cxcywh format."""
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target=None):
        image = F.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image, None
        target = target.copy()
        h, w = image.shape[-2:]
        if "boxes" in target:
            boxes = target["boxes"]
            boxes = box_xyxy_to_cxcywh(boxes)
            boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
            target["boxes"] = boxes
        return image, target


class Compose(object):
    """Compose multiple transforms."""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string

def make_data_transforms(image_set, args=None):
    """
    Create transforms for fire/smoke detection dataset.
    Minimal augmentation - only essential preprocessing.
    
    Args:
        image_set: 'train', 'val', or 'test'
        args: Optional arguments with image size settings
    
    Returns:
        Compose object with transforms
    """
    # Default ImageNet normalization (used by ResNet backbone)
    normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    # Get image size from args or use defaults
    if args is not None:
        scales = getattr(args, 'scales', [800])
        max_size = getattr(args, 'max_size', 1333)
    else:
        scales = [800]
        max_size = 1333
    
    if image_set == 'train':
        # Training: minimal augmentation (multi-scale resize only)
        return Compose([
            RandomResize(scales, max_size=max_size),
            ToTensor(),
            normalize,
        ])
    
    if image_set in ['val', 'test']:
        # Validation/Test: fixed resize, no augmentation
        return Compose([
            RandomResize([scales[0]], max_size=max_size),  # Use first scale only
            ToTensor(),
            normalize,
        ])
    
    raise ValueError(f'Unknown image_set: {image_set}')
