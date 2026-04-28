"""
Script to pre-compute SLIC superpixels (single or multi-scale).
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


def process_single_image(args_tuple):
    """
    Process a single image - designed for multiprocessing.
    Optimized for speed and memory efficiency on large-scale processing.
    
    Args:
        args_tuple: (img_path, images_dir, output_dir, scales, compactness, multiscale)
    
    Returns:
        (img_path, success, error_msg)
    """
    img_path, images_dir, output_dir, scales, compactness, multiscale = args_tuple
    
    try:
        # Import here to avoid pickling issues in multiprocessing
        from skimage.segmentation import slic
        from skimage.util import img_as_float
        
        # Determine relative path to maintain directory structure
        rel_path = img_path.relative_to(images_dir)
        save_dir = output_dir / rel_path.parent
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Load and convert image once (in-place conversion to save memory)
        img = Image.open(img_path).convert('RGB')
        img_np = np.array(img, dtype=np.uint8)
        img_float = img_as_float(img_np)
        
        # Get image dimensions for logging if needed
        height, width = img_np.shape[:2]
        
        # Generate superpixels for each scale
        for n_sp in scales:
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
            
            # Run SLIC segmentation (CPU-based, optimized in scikit-image)
            # channel_axis=2 specifies that color channels are on axis 2
            sp_map = slic(
                img_float, 
                n_segments=n_sp, 
                compactness=compactness,
                start_label=0,
                channel_axis=2,
                enforce_connectivity=True,  # Ensures connected superpixels (adds ~10% computation)
                slic_zero=False  # Use default sigma (good balance between speed and quality)
            )
            
            # Convert to efficient numpy type (uint16 supports up to ~65k superpixels)
            sp_map_uint16 = sp_map.astype(np.uint16)
            
            # Save as compressed numpy array (significant storage savings)
            np.savez_compressed(save_path_npz, sp_map=sp_map_uint16)
            
            # Explicitly delete to free memory in this process
            del sp_map, sp_map_uint16
        
        return (img_path, True, None)
        
    except Exception as e:
        return (img_path, False, str(e))


def generate_superpixel(data_path, output_path, n_superpixels=150, compactness=10.0, 
                        multiscale=False, superpixel_scales=None, num_workers=None,
                        image_extensions=None, max_workers_cap=None):
    """
    Generate superpixel maps for all images using parallel processing.
    Fully standalone - no config file required.
    
    Args:
        data_path: Root path of dataset (must contain 'images' subdirectory)
        output_path: Output path for superpixels
        n_superpixels: Number of superpixels for single-scale (default: 150)
        compactness: SLIC compactness parameter (default: 10.0)
        multiscale: If True, generate maps for multiple scales
        superpixel_scales: List of scales to generate if multiscale=True (default: [150, 300, 600])
        num_workers: Number of parallel workers (default: auto-calculated based on CPU count)
        image_extensions: List of image extensions to search for (default: common formats)
        max_workers_cap: Maximum cap on workers even if CPU count is higher (default: 32)
    """
    
    # Setup paths
    data_root = Path(data_path)
    images_dir = data_root / 'images'
    output_dir = Path(output_path)
    
    if not images_dir.exists():
        print(f"❌ Error: Images directory not found at {images_dir}")
        print(f"   Expected structure: {data_root}/images/")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set defaults
    if image_extensions is None:
        image_extensions = ['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'tif']
    
    if superpixel_scales is None:
        superpixel_scales = [150, 300, 600]
    
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
        # Case-insensitive search
        image_files.extend(list(images_dir.rglob(f'*.{ext}')))
        image_files.extend(list(images_dir.rglob(f'*.{ext.upper()}')))
    
    # Remove duplicates
    image_files = list(set(image_files))
    image_files.sort()
    
    if len(image_files) == 0:
        print(f"❌ No images found in {images_dir}")
        return
    
    print(f"✓ Found {len(image_files)} images")
    
    # Calculate optimal number of workers
    if num_workers is None:
        cpu_count = mp.cpu_count()
        # Smart worker calculation:
        # - Use most CPUs but cap to avoid excessive memory usage
        # - For >16 CPUs: use 80% of available CPUs
        # - For <=16 CPUs: use all available CPUs
        if cpu_count > 16:
            num_workers = max(8, int(cpu_count * 0.8))
        else:
            num_workers = cpu_count
        num_workers = min(num_workers, max_workers_cap)
    
    print(f"✓ Using {num_workers} parallel workers (available CPUs: {mp.cpu_count()})")
    
    # Calculate total work
    total_maps = len(image_files) * len(scales)
    print(f"✓ Parameters: n_superpixels={scales[0] if not multiscale else scales}, "
          f"compactness={compactness}")
    print(f"✓ Total superpixel maps to generate: {total_maps}")
    print(f"✓ Output directory: {output_dir}\n")
    
    # Prepare arguments for each image
    task_args = [
        (img_path, images_dir, output_dir, scales, compactness, multiscale)
        for img_path in image_files
    ]
    
    # Process images in parallel
    success_count = 0
    error_count = 0
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        futures = {executor.submit(process_single_image, args): args[0] for args in task_args}
        
        # Process results as they complete
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
    maps_per_second = total_maps / elapsed_time
    
    print(f"\n{'='*60}")
    print(f"✓ Superpixel generation complete!")
    print(f"{'='*60}")
    print(f"  ✓ Success: {success_count} images")
    print(f"  ✗ Errors: {error_count} images")
    print(f"  ⏱️  Time elapsed: {elapsed_time:.2f}s")
    print(f"  ⚡ Speed: {maps_per_second:.1f} maps/second")
    print(f"  📁 Output directory: {output_dir}")
    if multiscale:
        print(f"  📊 Generated scales: {scales}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate SLIC superpixels offline (parallelized, standalone, no config needed)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
USAGE EXAMPLES:
  # Basic usage - single scale with 150 superpixels:
  python generate_superpixel.py --data_path /path/to/data --output_path /path/to/output

  # Multi-scale generation (150, 300, 600):
  python generate_superpixel.py --data_path /path/to/data --output_path /path/to/output \\
    --multiscale --superpixel_scales 150 300 600

  # Large server with many CPUs (use 50 workers):
  python generate_superpixel.py --data_path /path/to/data --output_path /path/to/output \\
    --num_workers 50 --max_workers_cap 64

  # Custom compactness and superpixel count:
  python generate_superpixel.py --data_path /path/to/data --output_path /path/to/output \\
    --n_superpixels 300 --compactness 15.0

  # High-speed multi-scale on big server (e.g., 128 CPU cores):
  python generate_superpixel.py --data_path /path/to/data --output_path /path/to/output \\
    --multiscale --superpixel_scales 100 200 400 600 --num_workers 100 --max_workers_cap 128
        """
    )
    
    # Required arguments
    parser.add_argument('--data_path', required=True, 
                        help='Root path of dataset (must contain "images" subdirectory)')
    parser.add_argument('--output_path', required=True, 
                        help='Output path for superpixels (will be created if not exists)')
    
    # Superpixel configuration
    parser.add_argument('--n_superpixels', type=int, default=200,
                        help='Number of superpixels for single-scale mode (default: 150)')
    parser.add_argument('--compactness', type=float, default=10.0,
                        help='SLIC compactness parameter (default: 10.0). Higher = more regular shapes.')
    parser.add_argument('--multiscale', action='store_true',
                        help='Enable multi-scale superpixel generation')
    parser.add_argument('--superpixel_scales', type=int, nargs='+', 
                        default=[100, 200, 400],
                        help='Superpixel scales for multi-scale mode (default: 150 300 600)')
    
    # Performance tuning
    parser.add_argument('--num_workers', type=int, default=None,
                        help='Number of parallel workers (default: auto - uses CPU count intelligently). '
                             'For large servers, increase this (e.g., 32-128)')
    parser.add_argument('--max_workers_cap', type=int, default=48,
                        help='Maximum cap on number of workers to avoid excessive memory (default: 32). '
                             'For very large servers, increase to match available CPUs (e.g., 64, 128)')
    
    # Optional advanced parameters
    parser.add_argument('--image_extensions', type=str, nargs='+',
                        default=['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'tif'],
                        help='Image extensions to search for (default: jpg jpeg png bmp tiff tif)')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("SLIC Superpixel Generator (Standalone, Parallelized)")
    print("="*60)
    
    generate_superpixel(
        data_path=args.data_path,
        output_path=args.output_path,
        n_superpixels=args.n_superpixels,
        compactness=args.compactness,
        multiscale=args.multiscale,
        superpixel_scales=args.superpixel_scales,
        num_workers=args.num_workers,
        image_extensions=args.image_extensions,
        max_workers_cap=args.max_workers_cap
    )