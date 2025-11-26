"""
Utility to visualize a saved superpixel PNG file.
"""
import sys
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse

def visualize_saved_superpixel(sp_path):
    """
    Load a saved superpixel NPY and visualize it overlaid on the original image.
    """
    sp_path = Path(sp_path).resolve()
    if not sp_path.exists():
        print(f"Error: File not found at {sp_path}")
        return

    # Load the superpixel map
    if sp_path.suffix == '.npz':
        with np.load(sp_path) as data:
            sp_map = data['sp_map']
    else:
        sp_map = np.load(sp_path)
    
    try:
        # Replace 'superpixels' with 'images' in the path
        parts = list(sp_path.parts)
        if 'superpixels' in parts:
            idx = parts.index('superpixels')
            parts[idx] = 'images'
            
            base_img_path = Path(*parts).with_suffix('')
            
            # Check common extensions
            found_img = None
            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                img_path = base_img_path.with_suffix(ext)
                if img_path.exists():
                    found_img = img_path
                    break
            
            if found_img:
                print(f"Found original image: {found_img}")
                original_img = Image.open(found_img).convert('RGB')
                original_np = np.array(original_img)
            else:
                print(f"Warning: Original image not found for {base_img_path}")
                original_np = np.zeros((*sp_map.shape, 3), dtype=np.uint8)
        else:
             print("Path does not contain 'superpixels' folder, cannot deduce image path.")
             original_np = np.zeros((*sp_map.shape, 3), dtype=np.uint8)

    except Exception as e:
        print(f"Error finding image: {e}")
        original_np = np.zeros((*sp_map.shape, 3), dtype=np.uint8)

    print(f"Loaded {sp_path.name}")
    print(f"Number of Superpixels: {sp_map.max() + 1}")
    
    # Visualize
    from skimage.segmentation import mark_boundaries
    
    # Create visualization with boundaries
    viz = mark_boundaries(original_np, sp_map, color=(1, 1, 0)) # Yellow boundaries
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.imshow(viz)
    plt.title(f"Superpixel Segmentation: {sp_path.name}\n({sp_map.max() + 1} segments)")
    plt.axis('off')
    
    # Save preview
    output_dir = Path('outputs/superpixel_preview') 
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / f"overlay_{sp_path.stem}.png"
    
    plt.savefig(save_path)
    print(f"Saved visualization preview to: {save_path}")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help='Path to the superpixel NPY file')
    args = parser.parse_args()
    
    visualize_saved_superpixel(args.path)
