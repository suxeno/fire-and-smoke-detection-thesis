# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
from pathlib import Path

import torch
import torch.utils.data
import torchvision
from pycocotools import mask as coco_mask

import datasets.transforms as T


import os
import numpy as np


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(
        self,
        img_folder,
        ann_file,
        transforms,
        return_masks,
        superpixel_paths=None,
        require_superpixels=False,
    ):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)
        self.superpixel_paths = superpixel_paths or {}
        self.require_superpixels = require_superpixels

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)
        
        if self.superpixel_paths:
            slic_maps = {}
            img_info = self.coco.loadImgs(image_id)[0]
            rel_path = Path(img_info['file_name'])
            for n_seg, sp_dir in self.superpixel_paths.items():
                sp_path = sp_dir / rel_path.parent / (rel_path.stem + '.npz')
                if sp_path.exists():
                    with np.load(str(sp_path)) as data:
                        sp_map = torch.from_numpy(
                            data['sp_map'].astype(np.int64)
                        ).long()
                        # IMPORTANT (performance): do NOT compact superpixel IDs here.
                        # Dataset-level compaction is full-resolution and can be very expensive.
                        # The model can optionally compact IDs later on the *downsampled* map.
                        invalid_mask = (sp_map < 0) | (sp_map >= n_seg)
                        sp_map[invalid_mask] = -1
                        slic_maps[n_seg] = sp_map
                elif self.require_superpixels:
                    raise FileNotFoundError(f"Missing superpixel map for image '{rel_path}': {sp_path}")
            if slic_maps:
                target['slic_maps'] = slic_maps

        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


def make_coco_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])

    if image_set == 'test':
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    root = Path(args.coco_path)
    assert root.exists(), f'provided COCO path {root} does not exist'
    mode = 'instances'
    # PATHS = {
    #     "train": (root / "train2017", root / "annotations" / f'{mode}_train2017.json'),
    #     "val": (root / "val2017", root / "annotations" / f'{mode}_val2017.json'),
    # }

    # Auto-detect the COCO annotation folder
    # Structure is: root/annotations/COCO_*/Annotations/*.json
    coco_folder = None
    annotations_root = root / "annotations"
    if annotations_root.exists():
        for item in annotations_root.iterdir():
            if item.is_dir() and item.name.startswith("COCO_"):
                coco_folder = item.name
                break
    
    if coco_folder is None:
        # Fallback for standard COCO or if detection fails
        print(f"Warning: Could not auto-detect COCO_* folder in {annotations_root}. Using default structure.")
        # Standard COCO structure usually doesn't have the intermediate COCO_* folder, 
        # but for FASDD we fallback to the one found or assume standard
        
        # If standard coco: root/annotations/instances_train2017.json
        # If FASDD but failed: try to find any json in annotations/Annotations/
        pass

    # FASDD Structure 
    if coco_folder:
        PATHS = {
            "train": (
                root / "images",
                root / "annotations" / coco_folder / "Annotations" / "train.json",
            ),
            "val": (
                root / "images",
                root / "annotations" / coco_folder / "Annotations" / "val.json",
            ),
             "test": (
                root / "images",
                root / "annotations" / coco_folder / "Annotations" / "test.json",
            ),
        }
    else:
        # Fallback/Standard COCO
        PATHS = {
            "train": (root / "train2017", root / "annotations" / f'{mode}_train2017.json'),
            "val": (root / "val2017", root / "annotations" / f'{mode}_val2017.json'),
        }


    # Resolve pre-computed superpixel map directories
    superpixel_paths = {}
    n_seg = getattr(args, 'slic_n_segments', 200)
    require_superpixels = getattr(args, 'require_superpixels', False)
    sp_dir = root / f'superpixels-{n_seg}'
    if sp_dir.exists():
        superpixel_paths[n_seg] = sp_dir
    else:
        msg = f"Superpixel dir not found: {sp_dir}."
        if require_superpixels:
            raise FileNotFoundError(msg)
        print(f"Warning: {msg}")

    img_folder, ann_file = PATHS[image_set]
    dataset = CocoDetection(
        img_folder,
        ann_file,
        transforms=make_coco_transforms(image_set),
        return_masks=args.masks,
        superpixel_paths=superpixel_paths,
        require_superpixels=require_superpixels,
    )
    return dataset
