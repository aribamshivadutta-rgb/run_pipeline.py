import cv2
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# --- 1. DYNAMIC PATH DISCOVERY ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

RAW_BASE_DIR = os.path.join(DATA_DIR, 'raw', 'RxHandBD-Raw')
IMAGES_DIR = os.path.join(RAW_BASE_DIR, 'Images')
CLEAN_OUTPUT_DIR = os.path.join(DATA_DIR, 'clean', 'MedicalCRNN_clean')

possible_names = [
    'Prescription_Labels.xlsx - Sheet1.csv',
    'Prescription_Labels.csv',
    'Prescription_Labels.xlsx'
]

search_folders = [RAW_BASE_DIR, DATA_DIR, PROJECT_ROOT]

INPUT_CSV = None
for folder in search_folders:
    for name in possible_names:
        temp_path = os.path.join(folder, name)
        if os.path.exists(temp_path):
            INPUT_CSV = temp_path
            break
    if INPUT_CSV: break

TARGET_SIZE = (256, 64)


# --- 2. IMAGE CLEANING LOGIC ---
def clean_and_standardize(img_path, target_size):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None: return None
    h, w = img.shape
    tw, th = target_size
    scale = min(tw / w, th / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((th, tw), 255, dtype=np.uint8)
    y_off, x_off = (th - nh) // 2, (tw - nw) // 2
    canvas[y_off:y_off + nh, x_off:x_off + nw] = resized
    return cv2.adaptiveThreshold(canvas, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)


# --- 3. PIPELINE EXECUTION ---
def run_pipeline():
    if not INPUT_CSV:
        print(f"CRITICAL ERROR: Label file not found.")
        return

    if not os.path.exists(IMAGES_DIR):
        print(f"CRITICAL ERROR: Images folder not found at: {IMAGES_DIR}")
        return

    if not os.path.exists(CLEAN_OUTPUT_DIR):
        os.makedirs(CLEAN_OUTPUT_DIR)

    print(f"SUCCESS: Found Labels at {INPUT_CSV}")

    # --- ULTRA-ROBUST DATA LOADING ---
    try:
        # Check if it's actually an Excel file despite the extension
        if INPUT_CSV.endswith('.xlsx'):
            df = pd.read_excel(INPUT_CSV)
        else:
            # Try reading as CSV, using on_bad_lines to skip messy rows if needed
            # and engine='python' to handle encoding more flexibly
            df = pd.read_csv(INPUT_CSV, encoding='ISO-8859-1', on_bad_lines='skip', engine='python')
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Clean the dataframe: Take only the first two columns (Images, Text)
    df = df.iloc[:, [0, 1]]
    df.columns = ['Images', 'Text']
    df = df.dropna()
    # ---------------------------------

    print(f"Loaded {len(df)} labels. Starting split and process...")
    train_df, test_df = train_test_split(df, test_size=0.20, random_state=42)

    train_df.to_csv(os.path.join(CLEAN_OUTPUT_DIR, 'Train_Label.csv'), index=False)
    test_df.to_csv(os.path.join(CLEAN_OUTPUT_DIR, 'Test_Label.csv'), index=False)

    for mode, data in {'train': train_df, 'test': test_df}.items():
        mode_path = os.path.join(CLEAN_OUTPUT_DIR, mode)
        if not os.path.exists(mode_path): os.makedirs(mode_path)

        print(f"\nProcessing {mode} set...")
        for img_name in tqdm(data['Images']):
            in_p = os.path.join(IMAGES_DIR, str(img_name))
            out_p = os.path.join(mode_path, str(img_name))

            if os.path.exists(in_p):
                processed_img = clean_and_standardize(in_p, TARGET_SIZE)
                if processed_img is not None:
                    cv2.imwrite(out_p, processed_img)

    print("\n" + "=" * 50)
    print(f"PIPELINE COMPLETE: Data saved to {CLEAN_OUTPUT_DIR}")
    print("=" * 50)


if __name__ == "__main__":
    run_pipeline()