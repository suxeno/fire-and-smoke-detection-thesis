import argparse
import json
import shutil
import random
import os
from pathlib import Path
from collections import Counter
import numpy as np
from tqdm import tqdm

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    
def get_distribution(annotations):
    """Calculate the distribution of categories in a list of annotations."""
    counts = Counter([ann['category_id'] for ann in annotations])
    total = sum(counts.values())
    if total == 0:
        return {}
    return {k: v / total for k, v in counts.items()}

def calculate_distance(dist1, dist2):
    """Calculate chi-squared distance between two distributions."""
    all_keys = set(dist1.keys()) | set(dist2.keys())
    distance = 0
    for k in all_keys:
        p = dist1.get(k, 0)
        q = dist2.get(k, 0)
        if p + q > 0:
            distance += ((p - q) ** 2) / (p + q)
    return distance

def sample_best_of_n(images, annotations_map, ratio, n_trials=10):
    """
    Perform N random samplings and pick the one that best matches the original label distribution.
    """
    if ratio >= 1.0:
        return images
    
    target_size = int(len(images) * ratio)
    if target_size == 0:
        print("Warning: Sample ratio resulted in 0 images. Selecting at least 1.")
        target_size = 1
        
    print(f"Sampling {target_size} images from {len(images)} total...")
    
    # Calculate original distribution
    all_annotations = []
    for img in images:
        all_annotations.extend(annotations_map.get(img['id'], []))
    original_dist = get_distribution(all_annotations)
    
    best_sample = None
    best_distance = float('inf')
    best_stats = None
    
    for _ in range(n_trials):
        # Random sample
        current_sample = random.sample(images, target_size)
        
        # Calculate sample distribution
        sample_annotations = []
        for img in current_sample:
            sample_annotations.extend(annotations_map.get(img['id'], []))
        sample_dist = get_distribution(sample_annotations)
        
        # Calculate distance
        dist = calculate_distance(original_dist, sample_dist)
        
        if dist < best_distance:
            best_distance = dist
            best_sample = current_sample
            best_stats = (original_dist, sample_dist)
            
    return best_sample, best_stats

def main():
    parser = argparse.ArgumentParser(description="Sample FASDD dataset while preserving distribution.")
    parser.add_argument("--dataset_root", type=str, default="/home/Media/Dataset/FASDD/", help="Path to FASDD root")
    parser.add_argument("--category", type=str, required=True, help="Dataset Category (e.g., FASDD_CV, FASDD_UAV)")
    parser.add_argument("--ratio", type=float, default=0.1, help="Sampling ratio (0.0 to 1.0)")
    parser.add_argument("--output_root", type=str, default=None, help="Output directory root")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--dry-run", action="store_true", help="Run without copying files")
    
    args = parser.parse_args()
    set_seed(args.seed)
    
    dataset_root = Path(args.dataset_root)
    category_dir = dataset_root / args.category
    
    if not category_dir.exists():
        print(f"Error: Category directory not found: {category_dir}")
        return
        
    # Auto-detect structure
    images_dir = category_dir / "images"
    
    # Try to find the correct annotation folder (e.g., COCO_CV, COCO_UAV)
    # Heuristic: Find folder starting with COCO_ inside annotations
    annotations_root = category_dir / "annotations"
    coco_folder = None
    if annotations_root.exists():
        for item in annotations_root.iterdir():
            if item.is_dir() and item.name.startswith("COCO_"):
                coco_folder = item / "Annotations"
                break
    
    if not coco_folder or not coco_folder.exists():
        print(f"Error: Could not automatically locate COCO annotations in {annotations_root}")
        print("Expected structure: annotations/COCO_*/Annotations/*.json")
        return

    print(f"Found images at: {images_dir}")
    print(f"Found annotations at: {coco_folder}")
    
    # Setup output directory
    if args.output_root:
        output_base = Path(args.output_root)
    else:
        output_base = Path(f"./sample_{args.category}")
    
    print(f"Output directory: {output_base}")
    if args.dry_run:
        print("!!! DRY RUN MODE - No files will be copied !!!")
    
    splits = ["train", "val", "test"]
    
    for split in splits:
        ann_file = coco_folder / f"{split}.json"
        if not ann_file.exists():
            print(f"Skipping {split}, file not found: {ann_file}")
            continue
            
        print(f"\nProcessing {split} set...")
        with open(ann_file, 'r') as f:
            coco_data = json.load(f)
            
        images = coco_data['images']
        annotations = coco_data['annotations']
        categories = coco_data['categories']
        
        # Map image_id to annotations for quick access
        ann_map = {}
        for ann in annotations:
            img_id = ann['image_id']
            if img_id not in ann_map:
                ann_map[img_id] = []
            ann_map[img_id].append(ann)
            
        # Perform sampling
        selected_images, (orig_dist, sample_dist) = sample_best_of_n(images, ann_map, args.ratio)
        
        # Print stats
        print(f"  Selected {len(selected_images)}/{len(images)} images ({args.ratio*100:.1f}%)")
        print("  Class Distribution Comparison (Top 5):")
        
        # Map category ids to names
        cat_names = {c['id']: c['name'] for c in categories}
        
        sorted_keys = sorted(orig_dist.keys(), key=lambda k: orig_dist[k], reverse=True)[:5]
        print(f"    {'Category':<15} | {'Original':<10} | {'Sampled':<10} | {'Diff':<10}")
        print(f"    {'-'*15} | {'-'*10} | {'-'*10} | {'-'*10}")
        for k in sorted_keys:
            name = cat_names.get(k, str(k))
            p = orig_dist.get(k, 0) * 100
            q = sample_dist.get(k, 0) * 100
            diff = q - p
            print(f"    {name:<15} | {p:6.2f}%    | {q:6.2f}%    | {diff:+6.2f}%")
            
        if args.dry_run:
            continue
            
        # Prepare output data
        selected_ids = set(img['id'] for img in selected_images)
        new_annotations = [ann for ann in annotations if ann['image_id'] in selected_ids]
        
        new_coco_data = {
            "info": coco_data.get("info", {}),
            "licenses": coco_data.get("licenses", []),
            "categories": categories,
            "images": selected_images,
            "annotations": new_annotations
        }
        
        # Create directories
        out_images_dir = output_base / "images"
        out_ann_dir = output_base / "annotations" / coco_folder.parent.name / "Annotations"
        out_images_dir.mkdir(parents=True, exist_ok=True)
        out_ann_dir.mkdir(parents=True, exist_ok=True)
        
        # Save JSON
        out_json_path = out_ann_dir / f"{split}.json"
        with open(out_json_path, 'w') as f:
            json.dump(new_coco_data, f)
        print(f"  Saved annotations to {out_json_path}")
        
        # Copy Images
        print(f"  Copying {len(selected_images)} images...")
        for img in tqdm(selected_images):
            src_path = images_dir / img['file_name']
            dst_path = out_images_dir / img['file_name']
            
            if src_path.exists():
                shutil.copy2(src_path, dst_path)
            else:
                print(f"Warning: Image not found {src_path}")

    print("\nDone!")

if __name__ == "__main__":
    main()
