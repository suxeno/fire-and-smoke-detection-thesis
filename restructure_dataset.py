"""
Dataset Restructuring Script
Reorganizes images from flat structure to hierarchical: category/label/image.jpg
"""
import shutil
from pathlib import Path
from collections import defaultdict
import argparse
from tqdm import tqdm


def parse_filename(filename):
    """
    Extract label and category from filename.
    Example: bothFireAndSmoke_CV000000.jpg -> label='bothFireAndSmoke', category='CV'
    """
    stem = Path(filename).stem
    parts = stem.split('_')
    
    if len(parts) >= 2:
        label = parts[0]
        category_info = parts[1]
        category = ''.join([c for c in category_info if not c.isdigit()])
        return label, category
    
    return None, None


def restructure_images(source_dir, dest_dir, copy_files=True, verbose=True):
    """
    Restructure images from flat directory to category/label hierarchy.
    
    Args:
        source_dir: Source directory with flat image structure
        dest_dir: Destination directory for hierarchical structure
        copy_files: If True, copy files; if False, move files
        verbose: Print progress
    """
    source_path = Path(source_dir)
    dest_path = Path(dest_dir)
    
    # Get all image files
    image_extensions = ['*.jpg', '*.png', '*.tif', '*.jpeg', '*.JPG', '*.PNG', '*.TIF']
    image_files = []
    for ext in image_extensions:
        image_files.extend(source_path.glob(ext))
    
    if not image_files:
        print(f"⚠ No images found in {source_dir}")
        return
    
    # Count images by category and label
    stats = defaultdict(lambda: defaultdict(int))
    file_mapping = []
    
    # Parse all files first
    for img_file in image_files:
        label, category = parse_filename(img_file.name)
        
        if label and category:
            stats[category][label] += 1
            file_mapping.append((img_file, category, label))
    
    # Print statistics
    if verbose:
        print(f"\n{'='*60}")
        print(f"Restructuring Dataset: {source_dir}")
        print(f"{'='*60}\n")
        print(f"Total images found: {len(image_files)}\n")
        
        print("Distribution by Category and Label:")
        print(f"{'='*60}")
        for category in sorted(stats.keys()):
            print(f"\n{category}:")
            for label in sorted(stats[category].keys()):
                count = stats[category][label]
                print(f"  {label:25s}: {count:6d} images")
        print(f"\n{'='*60}\n")
    
    # Create directory structure and copy/move files
    operation = "Copying" if copy_files else "Moving"
    print(f"{operation} files to hierarchical structure...")
    
    processed = 0
    for img_file, category, label in tqdm(file_mapping, desc="Processing"):
        # Create destination directory: dest_dir/images/category/label/
        dest_category_label_dir = dest_path / category / label
        dest_category_label_dir.mkdir(parents=True, exist_ok=True)
        
        # Destination file path
        dest_file = dest_category_label_dir / img_file.name
        
        # Copy or move
        try:
            if copy_files:
                shutil.copy2(img_file, dest_file)
            else:
                shutil.move(str(img_file), str(dest_file))
            processed += 1
        except Exception as e:
            print(f"⚠ Error processing {img_file.name}: {e}")
    
    if verbose:
        print(f"\n✓ {operation} complete!")
        print(f"  Processed: {processed}/{len(file_mapping)} files")
        print(f"  Destination: {dest_path}")
        print(f"\n{'='*60}\n")
    
    return stats


def update_coco_annotations(annotations_dir, new_images_root):
    """
    Update COCO annotation file paths to match new hierarchical structure.
    
    Args:
        annotations_dir: Directory containing COCO annotation files
        new_images_root: New root path for images (e.g., 'datasets/images')
    """
    import json
    
    annotations_path = Path(annotations_dir)
    if not annotations_path.exists():
        print(f"⚠ Annotations directory not found: {annotations_dir}")
        return
    
    # Process each split
    for ann_file in annotations_path.glob('*.json'):
        print(f"\nUpdating {ann_file.name}...")
        
        with open(ann_file, 'r') as f:
            coco_data = json.load(f)
        
        # Update image file paths
        updated_count = 0
        for img in coco_data['images']:
            filename = img['file_name']
            
            # Extract label and category
            label, category = parse_filename(filename)
            
            if label and category:
                # Update file_name to new path: category/label/filename
                img['file_name'] = f"{category}/{label}/{Path(filename).name}"
                updated_count += 1
        
        # Save updated annotations
        with open(ann_file, 'w') as f:
            json.dump(coco_data, f, indent=2)
        
        print(f"  ✓ Updated {updated_count} image paths in {ann_file.name}")


def main():
    parser = argparse.ArgumentParser(
        description='Restructure dataset from flat to hierarchical (category/label) structure'
    )
    parser.add_argument('--source', type=str, required=True,
                       help='Source images directory (flat structure)')
    parser.add_argument('--dest', type=str, required=True,
                       help='Destination images directory (hierarchical structure)')
    parser.add_argument('--move', action='store_true',
                       help='Move files instead of copying (default: copy)')
    parser.add_argument('--update_annotations', type=str, default=None,
                       help='Path to COCO annotations directory to update')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress verbose output')
    
    args = parser.parse_args()
    
    # Restructure images
    restructure_images(
        source_dir=args.source,
        dest_dir=args.dest,
        copy_files=not args.move,
        verbose=not args.quiet
    )
    
    # Update COCO annotations if specified
    if args.update_annotations:
        update_coco_annotations(args.update_annotations, args.dest)
        print("\n✓ COCO annotations updated!")


if __name__ == "__main__":
    # Example usage if run without arguments
    import sys
    
    if len(sys.argv) == 1:
        print("\n" + "="*60)
        print("Dataset Restructuring Tool")
        print("="*60)
        print("\nThis script reorganizes images from:")
        print("  datasets/images/bothFireAndSmoke_CV000000.jpg")
        print("\nTo:")
        print("  datasets/images/CV/bothFireAndSmoke/bothFireAndSmoke_CV000000.jpg")
        print("\n" + "="*60)
        print("\nUsage examples:")
        print("\n1. Restructure main dataset (copy):")
        print("   python restructure_dataset.py \\")
        print("       --source datasets/images \\")
        print("       --dest datasets/images_restructured \\")
        print("       --update_annotations datasets/annotations/COCO/Annotations")
        print("\n2. Restructure in place (move files):")
        print("   python restructure_dataset.py \\")
        print("       --source datasets/images \\")
        print("       --dest datasets/images \\")
        print("       --move")
        print("\n" + "="*60 + "\n")
    else:
        main()