"""
Data Sampling Script for Fire and Smoke Detection Dataset
Creates a stratified sample maintaining label and category proportions
PRESERVES original train/val/test splits from COCO annotations
"""
import pandas as pd
import shutil
from pathlib import Path
from collections import defaultdict
import json
import random
import argparse

# Set random seed for reproducibility
random.seed(42)


def load_coco_split_info(annotations_dir):
    """
    Load which images belong to which split from COCO annotations.
    
    Returns:
        dict: {filename: split} mapping
    """
    annotations_path = Path(annotations_dir)
    split_mapping = {}
    
    for split in ['train', 'val', 'test']:
        ann_file = annotations_path / f'{split}.json'
        
        if ann_file.exists():
            with open(ann_file, 'r') as f:
                coco_data = json.load(f)
            
            for img in coco_data['images']:
                # Get just the filename (handle both flat and hierarchical paths)
                filename = Path(img['file_name']).name
                split_mapping[filename] = split
    
    return split_mapping


def load_dataset_info(images_path, hierarchical=False, annotations_dir=None):
    """
    Load all images and extract label, category, and split information.
    
    Args:
        images_path: Path to images directory
        hierarchical: If True, expects category/label/image.jpg structure
        annotations_dir: Path to COCO annotations to determine splits
    """
    images_path = Path(images_path)
    data = []
    
    # Load split information from COCO annotations
    split_mapping = {}
    if annotations_dir:
        split_mapping = load_coco_split_info(annotations_dir)
        print(f"✓ Loaded split information for {len(split_mapping)} images")
    
    if hierarchical:
        # Hierarchical structure: images/category/label/image.jpg
        for category_dir in images_path.iterdir():
            if not category_dir.is_dir():
                continue
            
            category = category_dir.name
            
            for label_dir in category_dir.iterdir():
                if not label_dir.is_dir():
                    continue
                
                label = label_dir.name
                
                # Get all images in this label directory
                image_files = list(label_dir.glob('*.jpg')) + \
                            list(label_dir.glob('*.png')) + \
                            list(label_dir.glob('*.tif'))
                
                for img_file in image_files:
                    split = split_mapping.get(img_file.name, 'unknown')
                    
                    data.append({
                        'filename': img_file.name,
                        'filepath': str(img_file),
                        'relative_path': f"{category}/{label}/{img_file.name}",
                        'label': label,
                        'category': category,
                        'split': split
                    })
    else:
        # Flat structure: images/label_category000000.jpg
        image_files = list(images_path.glob('*.jpg')) + \
                      list(images_path.glob('*.png')) + \
                      list(images_path.glob('*.tif'))
        
        for img_file in image_files:
            filename = img_file.stem
            
            parts = filename.split('_')
            if len(parts) >= 2:
                label = parts[0]
                category_info = parts[1]
                category = ''.join([c for c in category_info if not c.isdigit()])
                split = split_mapping.get(img_file.name, 'unknown')
                
                data.append({
                    'filename': img_file.name,
                    'filepath': str(img_file),
                    'relative_path': img_file.name,
                    'label': label,
                    'category': category,
                    'split': split
                })
    
    return pd.DataFrame(data)


def print_dataset_statistics(df, title="Dataset Statistics"):
    """Print dataset statistics."""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    
    print("\n=== Summary by Label ===")
    print(df['label'].value_counts())
    
    print("\n=== Summary by Category ===")
    print(df['category'].value_counts())
    
    print("\n=== Summary by Label and Category ===")
    print(df.groupby(['label', 'category']).size().unstack(fill_value=0))
    
    if 'split' in df.columns:
        print("\n=== Summary by Split ===")
        print(df['split'].value_counts())
        
        print("\n=== Summary by Split and Label ===")
        print(df.groupby(['split', 'label']).size().unstack(fill_value=0))
    
    print(f"\n=== Total Images: {len(df)} ===\n")


def stratified_sample_per_split(df, n_samples, preserve_split_ratio=True):
    """
    Perform stratified sampling maintaining label, category, AND split proportions.
    
    Args:
        df: DataFrame with image information
        n_samples: Total number of samples to take
        preserve_split_ratio: If True, maintain original train/val/test ratios
    """
    if 'split' not in df.columns or df['split'].isna().all():
        print("⚠ Warning: No split information found, sampling without split preservation")
        return stratified_sample(df, n_samples)
    
    # Calculate split proportions
    split_counts = df['split'].value_counts()
    total_count = len(df)
    
    print(f"\n{'='*60}")
    print("Split Proportions in Original Dataset")
    print(f"{'='*60}\n")
    
    for split, count in split_counts.items():
        proportion = count / total_count
        print(f"{split:10s}: {count:6d} images ({proportion*100:5.2f}%)")
    
    # Sample from each split separately
    sampled_dfs = []
    
    print(f"\n{'='*60}")
    print("Stratified Sampling per Split")
    print(f"{'='*60}\n")
    
    for split in ['train', 'val', 'test']:
        split_df = df[df['split'] == split]
        
        if len(split_df) == 0:
            continue
        
        # Calculate samples for this split
        if preserve_split_ratio:
            split_proportion = len(split_df) / total_count
            split_n_samples = int(n_samples * split_proportion)
        else:
            split_n_samples = n_samples // 3  # Equal distribution
        
        print(f"\n--- {split.upper()} Split ---")
        print(f"Target samples: {split_n_samples}")
        
        # Stratified sampling within this split
        split_sampled = stratified_sample(split_df, split_n_samples, verbose=True)
        sampled_dfs.append(split_sampled)
    
    # Combine all splits
    sampled_df = pd.concat(sampled_dfs, ignore_index=True)
    
    # Adjust to exact target if needed
    if len(sampled_df) < n_samples:
        remaining = n_samples - len(sampled_df)
        additional = df[~df.index.isin(sampled_df.index)].sample(n=remaining, random_state=42)
        sampled_df = pd.concat([sampled_df, additional], ignore_index=True)
    elif len(sampled_df) > n_samples:
        sampled_df = sampled_df.sample(n=n_samples, random_state=42)
    
    return sampled_df


def stratified_sample(df, n_samples, verbose=False):
    """
    Perform stratified sampling maintaining label and category proportions.
    """
    total_count = len(df)
    
    # Group by both label and category
    grouped = df.groupby(['label', 'category'])
    
    sampled_dfs = []
    
    if verbose:
        print(f"\nStratified Sampling Details:")
    
    for (label, category), group in grouped:
        # Calculate proportion
        proportion = len(group) / total_count
        
        # Calculate number of samples for this group
        group_samples = int(n_samples * proportion)
        
        # Ensure at least 1 sample if group exists
        if group_samples == 0 and len(group) > 0:
            group_samples = 1
        
        # Sample from this group
        if len(group) <= group_samples:
            sampled_group = group
        else:
            sampled_group = group.sample(n=group_samples, random_state=42)
        
        sampled_dfs.append(sampled_group)
        
        if verbose:
            print(f"  {label:20s} | {category:5s} | "
                  f"Orig: {len(group):6d} | Sampled: {group_samples:4d} | "
                  f"Prop: {proportion*100:5.2f}%")
    
    # Combine all sampled groups
    sampled_df = pd.concat(sampled_dfs, ignore_index=True)
    
    # Adjust to target
    if len(sampled_df) < n_samples:
        remaining = n_samples - len(sampled_df)
        additional = df[~df.index.isin(sampled_df.index)].sample(n=remaining, random_state=42)
        sampled_df = pd.concat([sampled_df, additional], ignore_index=True)
    elif len(sampled_df) > n_samples:
        sampled_df = sampled_df.sample(n=n_samples, random_state=42)
    
    return sampled_df


def copy_images_hierarchical(sampled_df, dest_dir):
    """
    Copy sampled images to destination directory maintaining hierarchical structure.
    Structure: dest_dir/category/label/image.jpg
    """
    dest_dir = Path(dest_dir)
    
    copied_count = 0
    for _, row in sampled_df.iterrows():
        src_path = Path(row['filepath'])
        
        # Create hierarchical path: category/label/filename
        dest_path = dest_dir / row['category'] / row['label'] / row['filename']
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        if src_path.exists():
            shutil.copy2(src_path, dest_path)
            copied_count += 1
    
    print(f"\n✓ Copied {copied_count} images to {dest_dir}")
    print(f"  Structure: category/label/image.jpg")
    return copied_count


def update_coco_annotations(sampled_df, original_ann_file, output_ann_file, split='train'):
    """Update COCO annotation file to only include sampled images."""
    if not Path(original_ann_file).exists():
        print(f"⚠ Warning: {original_ann_file} not found, skipping")
        return
    
    with open(original_ann_file, 'r') as f:
        coco_data = json.load(f)
    
    # Get sampled filenames for this split
    split_df = sampled_df[sampled_df['split'] == split]
    sampled_filenames = set(split_df['filename'].tolist())
    
    if len(sampled_filenames) == 0:
        print(f"⚠ No images for split '{split}', skipping")
        return
    
    # Filter images
    new_images = []
    for img in coco_data['images']:
        file_name = img['file_name']
        just_filename = Path(file_name).name
        
        if just_filename in sampled_filenames:
            # Update to hierarchical path
            matching_row = split_df[split_df['filename'] == just_filename]
            
            if not matching_row.empty:
                row = matching_row.iloc[0]
                img['file_name'] = f"{row['category']}/{row['label']}/{row['filename']}"
                new_images.append(img)
    
    # Get image IDs
    sampled_image_ids = {img['id'] for img in new_images}
    
    # Filter annotations
    new_annotations = [ann for ann in coco_data['annotations'] 
                       if ann['image_id'] in sampled_image_ids]
    
    # Create new COCO data
    new_coco_data = {
        'images': new_images,
        'annotations': new_annotations,
        'categories': coco_data['categories']
    }
    
    # Save
    output_ann_file = Path(output_ann_file)
    output_ann_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_ann_file, 'w') as f:
        json.dump(new_coco_data, f, indent=2)
    
    print(f"\n✓ Updated COCO annotations: {output_ann_file.name}")
    print(f"  - Images: {len(new_images)}")
    print(f"  - Annotations: {len(new_annotations)}")


def create_sample_dataset(n_samples=2000, 
                         images_dir='datasets/images',
                         annotations_dir='datasets/annotations/COCO/Annotations',
                         output_dir='datasets_sample',
                         hierarchical=False,
                         preserve_splits=True):
    """
    Main function to create sampled dataset PRESERVING original train/val/test splits.
    """
    print(f"\n{'='*60}")
    print(f"Creating Sampled Dataset: {n_samples} images")
    print(f"Structure: {'Hierarchical (category/label)' if hierarchical else 'Flat'}")
    print(f"Preserve Splits: {preserve_splits}")
    print(f"{'='*60}\n")
    
    images_path = Path(images_dir)
    
    # Load full dataset with split information
    print("Loading full dataset with split information...")
    df = load_dataset_info(images_path, hierarchical=hierarchical, 
                          annotations_dir=annotations_dir if preserve_splits else None)
    print_dataset_statistics(df, "Original Dataset Statistics")
    
    # Perform stratified sampling per split
    print("\nPerforming stratified sampling...")
    if preserve_splits and 'split' in df.columns:
        sampled_df = stratified_sample_per_split(df, n_samples, preserve_split_ratio=True)
    else:
        sampled_df = stratified_sample(df, n_samples, verbose=True)
    
    # Print sampled statistics
    print_dataset_statistics(sampled_df, "Sampled Dataset Statistics")
    
    # Copy images with hierarchical structure
    output_images_dir = Path(output_dir) / 'images'
    copy_images_hierarchical(sampled_df, output_images_dir)
    
    # Update COCO annotations per split
    print(f"\n{'='*60}")
    print("Updating COCO Annotations")
    print(f"{'='*60}")
    
    for split in ['train', 'val', 'test']:
        original_ann = Path(annotations_dir) / f'{split}.json'
        output_ann = Path(output_dir) / 'annotations' / 'COCO' / 'Annotations' / f'{split}.json'
        
        if original_ann.exists():
            update_coco_annotations(sampled_df, original_ann, output_ann, split)
    
    # Save sampled filenames for reference
    output_csv = Path(output_dir) / 'sampled_files.csv'
    sampled_df.to_csv(output_csv, index=False)
    print(f"\n✓ Saved sampled file list to: {output_csv}")
    
    print(f"\n{'='*60}")
    print(f"✓ Sample dataset created successfully!")
    print(f"  Location: {output_dir}")
    print(f"  Structure: category/label/image.jpg")
    print(f"  Splits: Preserved from original dataset")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create stratified sample preserving train/val/test splits')
    parser.add_argument('--n_samples', type=int, default=2000,
                       help='Number of images to sample (default: 2000)')
    parser.add_argument('--images_dir', type=str, default='datasets/images',
                       help='Source images directory')
    parser.add_argument('--annotations_dir', type=str, default='datasets/annotations/COCO/Annotations',
                       help='Source COCO annotations directory')
    parser.add_argument('--output_dir', type=str, default='datasets_sample',
                       help='Output directory for sampled dataset')
    parser.add_argument('--hierarchical', action='store_true',
                       help='Source uses hierarchical structure (category/label/image.jpg)')
    parser.add_argument('--no_preserve_splits', action='store_true',
                       help='Do NOT preserve original train/val/test splits')
    
    args = parser.parse_args()
    
    create_sample_dataset(
        n_samples=args.n_samples,
        images_dir=args.images_dir,
        annotations_dir=args.annotations_dir,
        output_dir=args.output_dir,
        hierarchical=args.hierarchical,
        preserve_splits=not args.no_preserve_splits
    )