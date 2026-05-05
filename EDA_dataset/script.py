import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import imagesize
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

# --- CONFIGURATION ---
IMG_DIR = "/home/master/Documents/TA/datasets/images/CV"
ANN_DIR = "/home/master/Documents/TA/datasets/annotations/YOLO/labels"
SAVE_DIR = "./eda_results"
CLASS_NAMES = {0: "Fire", 1: "Smoke"}

# Set this to True if you want to force the script to run even if files exist
OVERWRITE = False

# List of files to check for skip logic
REQUIRED_PLOTS = [
    "0_folder_distribution.png",
    "1_object_distribution.png",
    "2_bbox_sizes.png",
    "3_spatial_heatmap.png"
]

os.makedirs(SAVE_DIR, exist_ok=True)

def check_existing_results():
    """Returns True if all plots already exist in the SAVE_DIR."""
    if OVERWRITE:
        return False
    
    existing_files = os.listdir(SAVE_DIR)
    missing = [f for f in REQUIRED_PLOTS if f not in existing_files]
    
    if not missing:
        print(f"[*] EDA results already exist in {SAVE_DIR}. Skipping processing.")
        print("[!] Set OVERWRITE = True in the script if you want to re-run.")
        return True
    return False

def process_image_task(img_info):
    img_path, category = img_info
    try:
        filename = os.path.splitext(os.path.basename(img_path))[0]
        ann_path = os.path.join(ANN_DIR, f"{filename}.txt")
        
        w, h = imagesize.get(img_path)
        
        bboxes = []
        if os.path.exists(ann_path):
            with open(ann_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5: continue
                    
                    cls_id = int(parts[0])
                    xc_norm, yc_norm = float(parts[1]), float(parts[2])
                    bw_norm, bh_norm = float(parts[3]), float(parts[4])
                    
                    bboxes.append({
                        "obj_class": CLASS_NAMES.get(cls_id, f"ID_{cls_id}"),
                        "bbox_area_ratio": bw_norm * bh_norm,
                        "x_center": xc_norm * w,
                        "y_center": yc_norm * h
                    })
        
        return {"category": category, "img_w": w, "img_h": h, "bboxes": bboxes}
    except Exception:
        return None

def run_fast_eda():
    # 1. Skip Logic Check
    if check_existing_results():
        return

    # 2. Indexing
    print("Indexing files and folders...")
    tasks = []
    for folder in os.listdir(IMG_DIR):
        folder_path = os.path.join(IMG_DIR, folder)
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    tasks.append((os.path.join(folder_path, file), folder))
    
    if not tasks:
        print("No images found. Check IMG_DIR.")
        return

    # 3. Parallel Processing
    print(f"Processing {len(tasks)} images...")
    num_workers = multiprocessing.cpu_count()
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(executor.map(process_image_task, tasks), 
                            total=len(tasks), 
                            desc="Analyzing Dataset"))

    # 4. Data Aggregation
    image_data = []
    object_data = []

    for r in results:
        if r:
            image_data.append({"category": r['category'], "img_w": r['img_w'], "img_h": r['img_h']})
            for bbox in r['bboxes']:
                object_data.append(bbox)

    img_df = pd.DataFrame(image_data)
    obj_df = pd.DataFrame(object_data)
    
    # 5. Visualizations
    print("Generating visualizations...")
    sns.set_theme(style="whitegrid")
    
    # Plot 0: Folder Dist
    plt.figure(figsize=(12, 6))
    order = img_df['category'].value_counts().index
    ax = sns.countplot(data=img_df, y='category', order=order, palette='viridis')
    ax.bar_label(ax.containers[0], padding=3)
    plt.title("Image distribution by Class")
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/0_folder_distribution.png")

    if not obj_df.empty:
        # Plot 1: Object Dist
        plt.figure(figsize=(10, 5))
        ax = sns.countplot(data=obj_df, x='obj_class', palette='flare')
        ax.bar_label(ax.containers[0], padding=3)
        plt.title("Object Instance Distribution")
        plt.savefig(f"{SAVE_DIR}/1_object_distribution.png")

        # Plot 2: BBox Sizes
        plt.figure(figsize=(10, 5))
        sns.kdeplot(data=obj_df, x="bbox_area_ratio", hue="obj_class", fill=True)
        plt.title("BBox Area Ratio Distribution")
        plt.xlim(0, 0.15) 
        plt.savefig(f"{SAVE_DIR}/2_bbox_sizes.png")

        # Plot 3: Heatmap
        plt.figure(figsize=(8, 7))
        plt.hexbin(obj_df['x_center'], obj_df['y_center'], gridsize=30, cmap='YlOrRd')
        plt.title("Spatial Heatmap of Objects")
        plt.gca().invert_yaxis()
        plt.savefig(f"{SAVE_DIR}/3_spatial_heatmap.png")

    print(f"\nDone! Results saved to: {os.path.abspath(SAVE_DIR)}")

if __name__ == "__main__":
    run_fast_eda()
