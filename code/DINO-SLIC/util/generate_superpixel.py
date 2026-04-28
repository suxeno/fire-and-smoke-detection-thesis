"""
Script to pre-compute SLIC superpixels (single or multi-scale).
Supports parallel processing for faster generation.
"""
import sys
import os
from pathlib import Path

# Add project root
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

from util import load_config


def process_single_image(args_tuple):
    """
    Process a single image - designed for multiprocessing.
    
    Args:
        args_tuple: (img_path, images_dir, output_dir, scales, compactness, multiscale)
    
    Returns:
        (img_path, success, error_msg)
    """
    img_path, images_dir, output_dir, scales, compactness, multiscale = args_tuple
    
    try:
        # Import here to avoid pickling issues
        from skimage.segmentation import slic
        from skimage.util import img_as_float
        
        # Determine relative path to maintain structure
        rel_path = img_path.relative_to(images_dir)
        save_dir = output_dir / rel_path.parent
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Load image once
        img = Image.open(img_path).convert('RGB')
        img_np = np.array(img)
        img_float = img_as_float(img_np)
        
        # Generate superpixels for each scale
        for n_sp in scales:
            if multiscale:
                save_path_npz = save_dir / f"{img_path.stem}_slic{n_sp}.npz"
            else:
                save_path_npz = save_dir / (img_path.stem + '.npz')
                save_path_npy = save_dir / (img_path.stem + '.npy')
                if save_path_npz.exists() or save_path_npy.exists():
                    continue
            
            if save_path_npz.exists():
                continue
            
            # Run SLIC using scikit-image (CPU, faster than custom implementation)
            sp_map = slic(
                img_float, 
                n_segments=n_sp, 
                compactness=compactness,
                start_label=0,
                channel_axis=2
            )
            
            # Save as compressed numpy array
            sp_map_np = sp_map.astype(np.uint16)
            np.savez_compressed(save_path_npz, sp_map=sp_map_np)
        
        return (img_path, True, None)
        
    except Exception as e:
        return (img_path, False, str(e))


def generate_superpixel(config_path, data_path, output_path, multiscale=False, num_workers=None):
    """
    Generate superpixel maps for all images using parallel processing.
    
    Args:
        config_path: Path to config file
        data_path: Root path of dataset
        output_path: Output path for superpixels
        multiscale: If True, generate maps for scales [150, 300, 600]
        num_workers: Number of parallel workers (default: CPU count)
    """
    
    # Load config
    args = load_config(config_path)
    
    if not hasattr(args, 'n_superpixels') or not hasattr(args, 'slic_compactness'):
        raise ValueError("Config file must specify 'n_superpixels' and 'slic_compactness'")

    compactness = args.slic_compactness
    
    # Determine scales to generate
    if multiscale:
        scales = getattr(args, 'superpixel_scales', [150, 300, 600])
        print(f"Multi-scale mode: generating superpixels for scales {scales}")
    else:
        scales = [args.n_superpixels]
    
    # Setup paths
    data_root = Path(data_path)
    images_dir = data_root / 'images'
    output_dir = Path(output_path)
    
    if not images_dir.exists():
        print(f"Error: Images directory not found at {images_dir}")
        return

    # Find all images
    print(f"Scanning for images in {images_dir}...")
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(images_dir.rglob(ext)))
    
    print(f"Found {len(image_files)} images.")
    
    # Determine number of workers
    if num_workers is None:
        num_workers = min(mp.cpu_count(), 16)  # Cap at 16 to avoid memory issues
    
    print(f"Using {num_workers} parallel workers")
    print(f"Generating {len(scales)} scale(s) per image: {scales}")
    print(f"Total superpixel maps to generate: {len(image_files) * len(scales)}")
    
    # Prepare arguments for each image
    task_args = [
        (img_path, images_dir, output_dir, scales, compactness, multiscale)
        for img_path in image_files
    ]
    
    # Process images in parallel
    success_count = 0
    error_count = 0
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        futures = {executor.submit(process_single_image, args): args[0] for args in task_args}
        
        # Process results as they complete
        with tqdm(total=len(image_files), desc="Generating Superpixels") as pbar:
            for future in as_completed(futures):
                img_path, success, error_msg = future.result()
                if success:
                    success_count += 1
                else:
                    error_count += 1
                    print(f"\nError processing {img_path}: {error_msg}")
                pbar.update(1)
    
    print(f"\nSuperpixel generation complete!")
    print(f"  Success: {success_count}")
    print(f"  Errors: {error_count}")
    print(f"  Output directory: {output_dir}")
    if multiscale:
        print(f"  Generated scales: {scales}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate SLIC superpixels offline (parallelized)")
    parser.add_argument('--config', default='configs/detr_slic.yaml', help='Path to config file')
    parser.add_argument('--data_path', default='datasets', help='Root path of dataset')
    parser.add_argument('--output_path', default='datasets/superpixels', help='Output path for superpixels')
    parser.add_argument('--multiscale', action='store_true', help='Generate multi-scale superpixels (150, 300, 600)')
    parser.add_argument('--num_workers', type=int, default=None, help='Number of parallel workers (default: CPU count)')
    
    args = parser.parse_args()
    
    # If data_path is not provided, read from config
    if args.data_path == 'datasets':
        from util import load_config
        config_args = load_config(args.config)
        if hasattr(config_args, 'data_path'):
            args.data_path = config_args.data_path
            # Adjust output path if data_path is different (e.g. datasets_sample)
            if args.output_path == 'datasets/superpixels' and 'sample' in args.data_path:
                args.output_path = str(Path(args.data_path) / 'superpixels')

    print(f"Data Path: {args.data_path}")
    print(f"Output Path: {args.output_path}")
    print(f"Multi-scale: {args.multiscale}")
    print(f"Num Workers: {args.num_workers or 'auto (CPU count)'}")
    
    generate_superpixel(args.config, args.data_path, args.output_path, args.multiscale, args.num_workers)