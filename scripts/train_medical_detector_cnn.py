import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import os
import cv2
import numpy as np
from tqdm import tqdm

# Import your native U-Net architecture file
# (Ensure your architecture file is named medical_detector_cnn.py and is in the same directory)
from medical_detector_cnn import MedicalDetectorCNN


# ====================================================================
# 1. OPTIMIZED DATASET PIPELINE
# ====================================================================
class DetectorDataset(Dataset):
    def __init__(self, root_dir, mode='train'):
        self.img_dir = os.path.join(root_dir, mode, 'images')
        self.mask_dir = os.path.join(root_dir, mode, 'masks')

        if not os.path.exists(self.img_dir):
            raise FileNotFoundError(f"⚠️ Critical Error: Missing images directory at {self.img_dir}")

        # Filter out background tracking structures (e.g., .DS_Store)
        self.files = [f for f in os.listdir(self.img_dir)
                      if f.lower().endswith('.png') and not f.startswith('.')]

        self.transform = T.Compose([
            T.ToTensor(),  # Maps structural intensities from [0-255] directly to [0.0-1.0]
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.files[idx])
        mask_path = os.path.join(self.mask_dir, self.files[idx])

        # Load clean grayscale maps
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Normalize boundaries and construct tensor matrix arrays
        image = self.transform(image)
        mask = self.transform(mask)

        return image, mask


# ====================================================================
# 2. RUNTIME CORE TRAINING EXECUTION
# ====================================================================
def train_detector():
    # --- AUTOMATIC PATH MAPPING ---
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))

    DATA_ROOT = os.path.join(PROJECT_ROOT, 'data', 'clean', 'detector_clean')
    MODEL_SAVE_PATH = os.path.normpath(os.path.join(PROJECT_ROOT, 'models', 'medical_detector.pth'))

    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

    # --- HYPERPARAMETERS ---
    BATCH_SIZE = 16  # Scaled up to maximize GPU VRAM throughput cleanly
    EPOCHS = 30  # Set for convergence on complex, noisy document variations
    LEARNING_RATE = 1e-4

    # Target execution engine optimization check
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize Datasets and Loaders
    try:
        train_ds = DetectorDataset(DATA_ROOT, mode='train')

        # --- GPU ENHANCEMENTS ---
        # pin_memory=True locks RAM spaces to establish ultra-fast direct staging pipelines to CUDA VRAM.
        # num_workers=2 handles non-blocking multi-threaded image processing loops concurrently.
        train_loader = DataLoader(
            train_ds,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=2,
            pin_memory=True if torch.cuda.is_available() else False
        )
    except Exception as e:
        print(f"❌ DATA LIFECYCLE ERROR: {e}\nPlease run your preprocessor or verify folder structures.")
        return

    # Initialize Model Structures
    model = MedicalDetectorCNN(n_channels=1, n_classes=1).to(DEVICE)

    # Swapped out basic BCELoss to handle unconstrained logits linearly without numerical clipping bugs
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Gradient Scaler dynamically controls structural floating points under low-precision constraints
    scaler = torch.amp.GradScaler('cuda' if DEVICE.type == 'cuda' else 'cpu')

    print(f"🏥 System Ready: Starting Multi-Modal Vision Processing Layer...")

    # --- FIXED: Accessing .type to safely capitalize the device name string ---
    print(f"Targeting Processing Core: [{DEVICE.type.upper()}]")
    if DEVICE.type == 'cuda':
        print(f"Active Hardware Matrix Acceleration: {torch.cuda.get_device_name(0)}")
    print(f"Total Preprocessed Training Images Loaded: {len(train_ds)}")
    print("-" * 60)

    # --- MAIN ENGINE TRAIN LOOP ---
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0

        loop = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{EPOCHS}]")
        for images, masks in loop:
            # Shift data blocks concurrently with background processor loops
            images = images.to(DEVICE, non_blocking=True)
            masks = masks.to(DEVICE, non_blocking=True)

            # set_to_none=True removes pointers rather than re-writing 0 matrices, optimizing RAM overhead
            optimizer.zero_grad(set_to_none=True)

            # --- AMP ACCELERATION ---
            # Autocast automatically compresses FP32 tensor evaluations into ultra-lightweight FP16 arrays
            with torch.amp.autocast(device_type=DEVICE.type):
                outputs = model(images)
                loss = criterion(outputs, masks)

            # Step and rescale operations to maintain stable backward passes
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_loss = epoch_loss / len(train_loader)
        print(f"✨ Epoch {epoch + 1} Complete. Cross-Entropy Loss Mean = {avg_loss:.4f}")

        # --- LIVE CHECKPOINT PROTECTION ---
        # Overwrites the checkpoint instantly on disk after every single epoch.
        # If your machine loses power, you will never lose your progress!
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"💾 Checkpoint Matrix Verified: {os.path.basename(MODEL_SAVE_PATH)} synchronized smoothly.")

    print(f"\n" + "=" * 60)
    print(f"🏆 SUCCESS: Structural U-Net Core Weights Saved Directly to Target Node:")
    print(f"👉 {MODEL_SAVE_PATH}")
    print(f"=" * 60)


if __name__ == "__main__":
    train_detector()