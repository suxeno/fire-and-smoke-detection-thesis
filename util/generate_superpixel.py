"""
Script to pre-compute SLIC superpixels.
"""
import sys
import os
from pathlib import Path
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from util import load_config
from models.superpixel.slic import SLICSuperpixel

def generate_superpixel(config_path, data_path, output_path):
    # Load config
    args = load_config(config_path)
    
    if not hasattr(args, 'n_superpixels') or not hasattr(args, 'slic_compactness'):
        raise ValueError("Config file must specify 'n_superpixels' and 'slic_compactness'")

    n_superpixels = args.n_superpixels
    compactness = args.slic_compactness
    
    print(f"Initializing SLIC with n_segments={n_superpixels}, compactness={compactness}")
    slic = SLICSuperpixel(
        n_segments=n_superpixels,
        compactness=compactness,
        differentiable=False
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    slic = slic.to(device)
    
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
    
    # Process images
    for img_path in tqdm(image_files, desc="Generating Superpixels"):
        # Determine relative path to maintain structure
        rel_path = img_path.relative_to(images_dir)
        save_dir = output_dir / rel_path.parent
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Check for both .npy (legacy) and .npz (new)
        save_path_npy = save_dir / (img_path.stem + '.npy')
        save_path_npz = save_dir / (img_path.stem + '.npz')
        
        # Skip if already exists
        if save_path_npy.exists() or save_path_npz.exists():
            continue
            
        try:
            # Load image
            img = Image.open(img_path).convert('RGB')
            img_np = np.array(img)
            img_tensor = torch.from_numpy(img_np).float() / 255.0
            img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
            img_tensor = img_tensor.to(device)
            
            # Run SLIC (on original resolution)
            with torch.no_grad():
                superpixel_map, _ = slic(img_tensor)
            
            # Save as compressed numpy array
            # Use uint16 since n_superpixels (e.g. 300) fits easily in 16 bits (max 65535)
            sp_map_np = superpixel_map.cpu().squeeze().numpy().astype(np.uint16)
            
            # Use savez_compressed for significant space savings
            np.savez_compressed(save_path_npz, sp_map=sp_map_np)
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate SLIC superpixels offline")
    parser.add_argument('--config', default='configs/detr_slic.yaml', help='Path to config file')
    parser.add_argument('--data_path', default='datasets', help='Root path of dataset')
    parser.add_argument('--output_path', default='datasets/superpixels', help='Output path for superpixels')
    
    args = parser.parse_args()
    
    # If data_path is not provided, read from config
    if args.data_path == 'datasets':
        config_args = load_config(args.config)
        if hasattr(config_args, 'data_path'):
            args.data_path = config_args.data_path
            # Adjust output path if data_path is different (e.g. datasets_sample)
            if args.output_path == 'datasets/superpixels' and 'sample' in args.data_path:
                args.output_path = str(Path(args.data_path) / 'superpixels')

    print(f"Data Path: {args.data_path}")
    print(f"Output Path: {args.output_path}")
    
    generate_superpixel(args.config, args.data_path, args.output_path)
