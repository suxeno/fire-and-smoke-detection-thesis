"""
Script to pre-compute SLIC superpixels (single or multi-scale) or generate visual previews.
Supports parallel processing for faster generation on high-compute servers.
"""
import sys
import os
from pathlib import Path
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import time
import random
from collections import defaultdict


def process_single_image(args_tuple):
    """
    Process a single image - designed for multiprocessing.
    Optimized for speed and memory efficiency on large-scale processing.
    """
    img_path, images_dir, output_dir, scales, compactness, multiscale, is_preview = args_tuple
    
    try:
        # Import here to avoid pickling issues in multiprocessing
        from skimage.segmentation import slic, mark_boundaries
        from skimage.util import img_as_float
        import matplotlib.pyplot as plt
        
        # Determine relative path to maintain directory structure
        rel_path = img_path.relative_to(images_dir)
        
        # Output directory logic
        save_dir = output_dir / rel_path.parent
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Load and convert image once
        img = Image.open(img_path).convert('RGB')
        img_np = np.array(img, dtype=np.uint8)
        img_float = img_as_float(img_np)
        
        # Generate superpixels for each scale
        for n_sp in scales:
            if is_preview:
                if multiscale:
                    save_path_png = save_dir / f"{img_path.stem}_slic{n_sp}.png"
                else:
                    save_path_png = save_dir / (img_path.stem + '.png')
                
                # Skip if already exists
                if save_path_png.exists():
                    continue
            else:
                if multiscale:
                    save_path_npz = save_dir / f"{img_path.stem}_slic{n_sp}.npz"
                else:
                    save_path_npz = save_dir / (img_path.stem + '.npz')
                    # Check both .npz and old .npy formats to avoid re-processing
                    save_path_npy = save_dir / (img_path.stem + '.npy')
                    if save_path_npz.exists() or save_path_npy.exists():
                        continue
                
                # Skip if already exists
                if save_path_npz.exists():
                    continue
            
            # Run SLIC segmentation
            sp_map = slic(
                img_float, 
                n_segments=n_sp, 
                compactness=compactness,
                start_label=0,
                channel_axis=2,
                enforce_connectivity=True,
                slic_zero=False
            )
            
            if is_preview:
                # Draw boundaries with red color (1, 0, 0)
                bound_img = mark_boundaries(img_float, sp_map, color=(1, 0, 0), mode='thick')
                bound_img_uint8 = (bound_img * 255).astype(np.uint8)
                Image.fromarray(bound_img_uint8).save(save_path_png)
            else:
                # Save as compressed numpy array for actual training
                sp_map_uint16 = sp_map.astype(np.uint16)
                np.savez_compressed(save_path_npz, sp_map=sp_map_uint16)
                del sp_map_uint16
            
            del sp_map
        
        return (img_path, True, None)
        
    except Exception as e:
        return (img_path, False, str(e))


def generate_superpixel(data_path, output_path, n_superpixels=200, compactness=10.0, 
                        multiscale=False, superpixel_scales=None, num_workers=None,
                        image_extensions=None, max_workers_cap=None,
                        preview_only=False, samples_per_class=5):
    """
    Generate superpixel maps for images using parallel processing.
    """
    # Setup paths
    data_root = Path(data_path)
    images_dir = data_root / 'images'
    output_dir = Path(output_path)
    
    # Check if images directory exists. If not, assume data_path is the images directory
    if not images_dir.exists():
        images_dir = data_root
        
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set defaults
    if image_extensions is None:
        image_extensions = ['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'tif']
    
    if superpixel_scales is None:
        superpixel_scales = [100, 200, 400]
    
    if max_workers_cap is None:
        max_workers_cap = 32
    
    # Determine scales to generate
    if multiscale:
        scales = superpixel_scales
        print(f"✓ Multi-scale mode: generating superpixels for scales {scales}")
    else:
        scales = [n_superpixels]
        print(f"✓ Single-scale mode: generating superpixels for scale {n_superpixels}")
    
    # Find all images
    print(f"\n🔍 Scanning for images in {images_dir}...")
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(images_dir.rglob(f'*.{ext}')))
        image_files.extend(list(images_dir.rglob(f'*.{ext.upper()}')))
    
    image_files = list(set(image_files))
    image_files.sort()
    
    if len(image_files) == 0:
        print(f"❌ No images found in {images_dir}")
        return
        
    # Class balancing logic for preview mode
    if preview_only:
        print(f"🎨 Preview mode activated. Sampling up to {samples_per_class} images per class.")
        class_dict = defaultdict(list)
        for img_path in image_files:
            # Assume parent folder is class name
            class_name = img_path.parent.name
            class_dict[class_name].append(img_path)
            
        sampled_files = []
        for class_name, files in class_dict.items():
            if len(files) > samples_per_class:
                sampled_files.extend(random.sample(files, samples_per_class))
            else:
                sampled_files.extend(files)
        image_files = sampled_files
        print(f"✓ Selected {len(image_files)} images for preview generation across {len(class_dict)} classes.")
    else:
        print(f"✓ Found {len(image_files)} images for full processing.")
    
    # Calculate optimal number of workers
    if num_workers is None:
        cpu_count = mp.cpu_count()
        if cpu_count > 16:
            num_workers = max(8, int(cpu_count * 0.8))
        else:
            num_workers = cpu_count
        num_workers = min(num_workers, max_workers_cap)
    
    # Scale down workers if processing very few images (e.g. preview mode)
    if len(image_files) < num_workers:
        num_workers = max(1, len(image_files))
        
    print(f"✓ Using {num_workers} parallel workers")
    
    # Prepare arguments for each image
    task_args = [
        (img_path, images_dir, output_dir, scales, compactness, multiscale, preview_only)
        for img_path in image_files
    ]
    
    # Process images in parallel
    success_count = 0
    error_count = 0
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_single_image, args): args[0] for args in task_args}
        
        with tqdm(total=len(image_files), desc="Generating Superpixels", unit="img") as pbar:
            for future in as_completed(futures):
                img_path, success, error_msg = future.result()
                if success:
                    success_count += 1
                else:
                    error_count += 1
                    print(f"\n⚠️  Error processing {img_path}: {error_msg}")
                pbar.update(1)
    
    elapsed_time = time.time() - start_time
    total_maps = len(image_files) * len(scales)
    maps_per_second = total_maps / elapsed_time if elapsed_time > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"✓ Superpixel generation complete!")
    print(f"{'='*60}")
    print(f"  ✓ Mode: {'Preview (PNG output)' if preview_only else 'Data (NPZ output)'}")
    print(f"  ✓ Success: {success_count} images")
    print(f"  ✗ Errors: {error_count} images")
    print(f"  ⏱️  Time elapsed: {elapsed_time:.2f}s")
    if not preview_only:
        print(f"  ⚡ Speed: {maps_per_second:.1f} maps/second")
    print(f"  📁 Output directory: {output_dir}")
    if multiscale:
        print(f"  📊 Generated scales: {scales}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate SLIC superpixels offline (parallelized, standalone) or preview them visually.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
USAGE EXAMPLES:
  # Basic usage - generate NPZ for training (200 superpixels):
  python generate_superpixel.py --data_path /path/to/images --output_path /path/to/npz_output

  # Preview mode - generate visual PNGs overlaid with boundaries (5 samples per class):
  python generate_superpixel.py --data_path /path/to/images --output_path /path/to/preview_output \\
    --preview_only --samples_per_class 5

  # Multi-scale generation for training (100, 200, 400):
  python generate_superpixel.py --data_path /path/to/images --output_path /path/to/npz_output \\
    --multiscale --superpixel_scales 100 200 400
        """
    )
    
    # Required arguments
    parser.add_argument('--data_path', required=True, 
                        help='Root path of dataset (must contain "images" subdirectory, or be the images directory itself)')
    parser.add_argument('--output_path', required=True, 
                        help='Output path for superpixels or previews (will be created automatically)')
    
    # Superpixel configuration
    parser.add_argument('--n_superpixels', type=int, default=200,
                        help='Number of superpixels for single-scale mode (default: 200)')
    parser.add_argument('--compactness', type=float, default=10.0,
                        help='SLIC compactness parameter (default: 10.0). Higher = more regular shapes.')
    parser.add_argument('--multiscale', action='store_true',
                        help='Enable multi-scale superpixel generation')
    parser.add_argument('--superpixel_scales', type=int, nargs='+', 
                        default=[100, 200, 400],
                        help='Superpixel scales for multi-scale mode (default: 100 200 400)')
    
    # Performance tuning
    parser.add_argument('--num_workers', type=int, default=None,
                        help='Number of parallel workers (default: auto - uses CPU count intelligently)')
    parser.add_argument('--max_workers_cap', type=int, default=48,
                        help='Maximum cap on number of workers to avoid excessive memory (default: 48)')
    
    # Preview mode arguments
    parser.add_argument('--preview_only', action='store_true',
                        help='If set, generates visual PNG previews overlaid on images instead of saving .npz data')
    parser.add_argument('--samples_per_class', type=int, default=5,
                        help='Number of random samples to pick per class folder when in preview_only mode (default: 5)')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("SLIC Superpixel Generator & Previewer")
    print("="*60)
    
    generate_superpixel(
        data_path=args.data_path,
        output_path=args.output_path,
        n_superpixels=args.n_superpixels,
        compactness=args.compactness,
        multiscale=args.multiscale,
        superpixel_scales=args.superpixel_scales,
        num_workers=args.num_workers,
        max_workers_cap=args.max_workers_cap,
        preview_only=args.preview_only,
        samples_per_class=args.samples_per_class
    )
