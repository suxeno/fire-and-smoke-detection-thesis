import os
import glob
import numpy as np

def check_max_id():
    print("--- Checking Maximum Superpixel IDs ---")
    slic_dir = '/home/Media/Dataset/FASDD/FASDD_CV/superpixels-400'
    if not os.path.exists(slic_dir):
        return

    npz_files = glob.glob(os.path.join(slic_dir, '*.npz'))
    print(f"Total .npz files: {len(npz_files)}")
    
    max_id_overall = -1
    files_exceeding_500 = 0
    
    import random
    random.shuffle(npz_files)
    
    for i, npz_path in enumerate(npz_files):
        with np.load(npz_path) as data:
            sp_map = data['sp_map']
            max_id = sp_map.max()
            if max_id > max_id_overall:
                max_id_overall = max_id
            if max_id >= 500:
                files_exceeding_500 += 1
                
        if i > 0 and i % 5000 == 0:
            print(f"Checked {i}/{len(npz_files)} files. Current overall max: {max_id_overall}, Files >= 500: {files_exceeding_500}")
            
    print(f"Final overall max ID in superpixels-400: {max_id_overall}")
    print(f"Total files with max ID >= 500: {files_exceeding_500}")

if __name__ == '__main__':
    check_max_id()
        
    img_files = glob.glob(os.path.join(img_dir, '*.jpg'))
    flat_images = []
    
    # We'll just sample the first 5000 images to save time
    import cv2
    for img_path in img_files[:5000]:
        img = cv2.imread(img_path)
        if img is None:
            continue
        # If standard deviation is 0, the image is a perfectly solid color
        if np.std(img) == 0:
            flat_images.append(os.path.basename(img_path))
            
    print(f"Checked 5000 images for flatness.")
    print(f"Flat (solid color) images found: {len(flat_images)}")
    if flat_images:
        print(f"Examples of flat images: {flat_images[:5]}")
        
    print("\n")

if __name__ == '__main__':
    check_missing_maps()
    check_flat_images()
