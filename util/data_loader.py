"""
Fire and Smoke Detection Dataset Loader
Supports COCO format annotations.
"""
import torch
import torch.utils.data
from pathlib import Path
from PIL import Image
import json

from .misc import collate_fn
from .data_transform import make_data_transforms

class DatasetsLoader(torch.utils.data.Dataset):
    def __init__(self, img_folder, ann_file, transforms=None, return_masks=False, filter_category=None, superpixel_path=None):
        """
        Args:
            img_folder: Path to images root directory (e.g., 'datasets/images')
            ann_file: Path to COCO annotation file (e.g., 'datasets/annotations/COCO/Annotations/train.json')
            transforms: Transformations to apply
            return_masks: Whether to return segmentation masks (not used for detection)
            filter_category: Optional category string to filter images (e.g., 'CV', 'UAV', 'RS')
            superpixel_path: Path to pre-computed superpixels (e.g., 'datasets/superpixels')
        """
        self.img_folder = Path(img_folder)
        self.ann_file = Path(ann_file)
        self._transforms = transforms
        self.return_masks = return_masks
        self.superpixel_path = Path(superpixel_path) if superpixel_path else None
        self.prepare = ConvertCocoPolysToMask(return_masks)
        
        # Load COCO annotations
        with open(self.ann_file, 'r') as f:
            self.coco_data = json.load(f)
        
        # Create mappings
        self.images = {img['id']: img for img in self.coco_data['images']}
        self.categories = {cat['id']: cat for cat in self.coco_data['categories']}
        
        # Filter images by category if specified
        if filter_category:
            print(f"Filtering dataset for category: {filter_category}")
            self.images = {
                k: v for k, v in self.images.items() 
                if v['file_name'].startswith(f"{filter_category}/")
            }
        
        # Group annotations by image_id
        self.img_to_anns = {}
        for ann in self.coco_data['annotations']:
            img_id = ann['image_id']
            if img_id in self.images:  # Only keep annotations for filtered images
                if img_id not in self.img_to_anns:
                    self.img_to_anns[img_id] = []
                self.img_to_anns[img_id].append(ann)
        
        # List of image IDs
        self.ids = list(sorted(self.images.keys()))
        
        print(f"Loaded {len(self.ids)} images from {self.ann_file}")
        print(f"Categories: {[cat['name'] for cat in self.categories.values()]}")
    
    def __getitem__(self, idx):
        """
        Args:
            idx: Index
        
        Returns:
            image: PIL Image
            target: dict with keys:
                - boxes: [N, 4] in xyxy format
                - labels: [N] class labels
                - image_id: image identifier
                - area: [N] bounding box areas
                - iscrowd: [N] is crowd annotation
                - orig_size: (H, W) original image size
                - size: (H, W) current image size
        """
        img_id = self.ids[idx]
        img_info = self.images[img_id]
        
        # Load image
        img_path = self.img_folder / img_info['file_name']
        img = Image.open(img_path).convert('RGB')
        
        # Get annotations for this image
        ann_ids = self.img_to_anns.get(img_id, [])
        
        # Prepare target
        target = {'image_id': img_id, 'annotations': ann_ids}
        img, target = self.prepare(img, target)
        
        # Load superpixel map
        if self.superpixel_path:
            rel_path = Path(img_info['file_name'])
            # Try loading compressed .npz first (new format)
            sp_path_npz = self.superpixel_path / rel_path.parent / (rel_path.stem + '.npz')
            sp_path_npy = self.superpixel_path / rel_path.parent / (rel_path.stem + '.npy')
            
            if sp_path_npz.exists():
                import numpy as np
                # Load compressed array
                with np.load(sp_path_npz) as data:
                    sp_map = data['sp_map']
                target['slic_map'] = torch.from_numpy(sp_map.astype(np.int64)).long()
            elif sp_path_npy.exists():
                import numpy as np
                sp_map = np.load(sp_path_npy)
                target['slic_map'] = torch.from_numpy(sp_map.astype(np.int64)).long()
        
        # Apply transforms
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        
        return img, target
    
    def __len__(self):
        return len(self.ids)
    
    def get_category_name(self, label_id):
        """Get category name from label ID."""
        return self.categories.get(label_id, {}).get('name', 'unknown')


class ConvertCocoPolysToMask(object):
    """Convert COCO annotations to DETR format."""
    
    def __init__(self, return_masks=False):
        self.return_masks = return_masks
    
    def __call__(self, image, target):
        """
        Args:
            image: PIL Image
            target: dict with 'image_id' and 'annotations'
        
        Returns:
            image: PIL Image (unchanged)
            target: dict with processed annotations
        """
        w, h = image.size
        image_id = target["image_id"]
        annotations = target["annotations"]
        
        # Filter out invalid annotations
        anno = [obj for obj in annotations if 'bbox' in obj and 'category_id' in obj]
        
        boxes = []
        classes = []
        area = []
        iscrowd = []
        
        for obj in anno:
            # COCO bbox format: [x, y, width, height]
            bbox = obj['bbox']
            # Convert to [x_min, y_min, x_max, y_max]
            bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
            boxes.append(bbox)
            classes.append(obj['category_id'])
            area.append(obj.get('area', bbox[2] * bbox[3]))
            iscrowd.append(obj.get('iscrowd', 0))
        
        # Convert to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4) if boxes else torch.zeros((0, 4), dtype=torch.float32)
        classes = torch.tensor(classes, dtype=torch.int64) if classes else torch.zeros((0,), dtype=torch.int64)
        area = torch.tensor(area, dtype=torch.float32) if area else torch.zeros((0,), dtype=torch.float32)
        iscrowd = torch.tensor(iscrowd, dtype=torch.int64) if iscrowd else torch.zeros((0,), dtype=torch.int64)
        
        # Clamp boxes to image boundaries
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)
        
        # Remove invalid boxes (where min >= max)
        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        area = area[keep]
        iscrowd = iscrowd[keep]
        
        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target["image_id"] = torch.tensor([image_id])
        target["area"] = area
        target["iscrowd"] = iscrowd
        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])
        
        return image, target


def build_dataset(image_set, args, filter_category=None):
    """
    Build fire/smoke detection dataset.
    
    Args:
        image_set: 'train', 'val', or 'test'
        args: Arguments with dataset configuration
            - data_path: Root path to dataset (e.g., 'datasets' or 'datasets_sample')
            - use_sample: Whether to use sample dataset
        filter_category: Optional category to filter (CV, UAV, RS)
    
    Returns:
        FireSmokeDetection dataset
    """
    
    # Determine dataset root
    if getattr(args, 'use_sample', False):
        root = Path(getattr(args, 'data_path', 'datasets_sample'))
    else:
        root = Path(getattr(args, 'data_path', 'datasets'))
    
    assert root.exists(), f"Dataset path does not exist: {root}"
    
    # Paths
    img_folder = root / 'images'
    ann_file = root / 'annotations' / 'COCO' / 'Annotations' / f'{image_set}.json'
    superpixel_path = root / 'superpixels'
    
    assert img_folder.exists(), f"Images folder not found: {img_folder}"
    assert ann_file.exists(), f"Annotation file not found: {ann_file}"
    
    # Create dataset
    dataset = DatasetsLoader(
        img_folder=img_folder,
        ann_file=ann_file,
        transforms=make_data_transforms(image_set, args),
        return_masks=False,
        filter_category=filter_category,
        superpixel_path=superpixel_path if superpixel_path.exists() else None
    )
    
    return dataset


def build_data_loader(image_set, args, filter_category=None):
    """
    Build DataLoader for fire/smoke detection.
    
    Args:
        image_set: 'train', 'val', or 'test'
        args: Arguments with dataloader configuration
            - batch_size: Batch size for training
            - num_workers: Number of data loading workers
        filter_category: Optional category to filter (CV, UAV, RS)
    
    Returns:
        torch.utils.data.DataLoader
    """
    
    dataset = build_dataset(image_set, args, filter_category)
    
    # DataLoader settings
    if image_set == 'train':
        batch_size = getattr(args, 'batch_size', 2)
        shuffle = True
        drop_last = True
    else:
        batch_size = 1
        shuffle = False
        drop_last = False
    
    num_workers = getattr(args, 'num_workers', 4)
    
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=drop_last,
        pin_memory=True
    )
    
    return data_loader
