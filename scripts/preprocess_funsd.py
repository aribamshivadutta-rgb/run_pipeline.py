import os
import json
import cv2
import numpy as np
from tqdm import tqdm

# --- 1. DYNAMIC PATH DISCOVERY ---
# This ensures that no matter where your project is, the scripts find the right folders.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))

# Raw data path from your RAR (dataset/dataset/...)
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw', 'dataset', 'dataset')

# Output folder for the cleaned detector data
CLEAN_OUT_DIR = os.path.join(PROJECT_ROOT, 'data', 'clean', 'detector_clean')

TARGET_SIZE = (512, 512)

def process_set(set_folder_name):
    """Processes 'training_data' or 'testing_data'"""
    input_path = os.path.join(RAW_DATA_DIR, set_folder_name)

    if not os.path.exists(input_path):
        print(f"Skipping: {set_folder_name} not found at {input_path}")
        return

    images_in = os.path.join(input_path, 'images')
    json_in = os.path.join(input_path, 'annotations')

    # Force subfolders to be lowercase 'train' and 'test'
    mode = 'train' if 'training' in set_folder_name.lower() else 'test'
    img_out = os.path.join(CLEAN_OUT_DIR, mode, 'images')
    mask_out = os.path.join(CLEAN_OUT_DIR, mode, 'masks')

    os.makedirs(img_out, exist_ok=True)
    os.makedirs(mask_out, exist_ok=True)

    print(f"\nProcessing {set_folder_name} -> saving to lowercase '{mode}' folder...")

    # Filter for valid images and ignore hidden system files
    valid_ext = ('.png', '.jpg', '.jpeg')
    file_list = [f for f in os.listdir(images_in)
                 if f.lower().endswith(valid_ext) and not f.startswith('.')]

    for filename in tqdm(file_list):
        # 1. Load Image
        img_p = os.path.join(images_in, filename)
        img = cv2.imread(img_p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        h, w = img.shape

        # 2. Create Mask
        mask = np.zeros((h, w), dtype=np.uint8)

        # 3. Handle JSON Labels (with UTF-8 fix)
        file_base = os.path.splitext(filename)[0]
        json_p = os.path.join(json_in, file_base + '.json')

        if os.path.exists(json_p):
            # Added encoding='utf-8' to prevent UnicodeDecodeError on Windows
            with open(json_p, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                    for entity in data['form']:
                        box = entity['box']  # [x1, y1, x2, y2]
                        # Draw filled white rectangles on the black mask
                        cv2.rectangle(mask, (box[0], box[1]), (box[2], box[3]), 255, -1)
                except json.JSONDecodeError:
                    print(f"Error: Could not parse JSON for {filename}")

        # 4. Resize (512x512)
        # INTER_AREA for images (preserves detail)
        # INTER_NEAREST for masks (keeps binary values 0 or 255)
        img_res = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_AREA)
        mask_res = cv2.resize(mask, TARGET_SIZE, interpolation=cv2.INTER_NEAREST)

        # 5. Save with lowercase filename
        clean_filename = filename.lower()
        cv2.imwrite(os.path.join(img_out, clean_filename), img_res)
        cv2.imwrite(os.path.join(mask_out, clean_filename), mask_res)

def run_preprocessing():
    print("Starting FUNSD Preprocessing (UTF-8 Fixed Version)...")
    process_set('training_data')
    process_set('testing_data')
    print(f"\nSUCCESS: Clean detector data is ready in: {CLEAN_OUT_DIR}")

if __name__ == "__main__":
    run_preprocessing()