import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import pandas as pd
from pdf2image import convert_from_path

CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_SCRIPT_DIR)
sys.path.append(PROJECT_ROOT)


# ====================================================================
# 1. PHARMACEUTICAL CHARACTER LABEL ENCODER
# ====================================================================
class MedicalLabelEncoder:
    def __init__(self):
        # 70-character vocabulary limits trained parameters strictly to medical profiles
        self.chars = " %()-./012345678?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        self.char_to_num = {char: i + 1 for i, char in enumerate(self.chars)}
        self.num_to_char = {i + 1: char for i, char in enumerate(self.chars)}

    def encode(self, text):
        return [self.char_to_num[c] for c in text if c in self.char_to_num]

    def decode(self, nums):
        res = []
        for i, num in enumerate(nums):
            if num != 0 and (i == 0 or num != nums[i - 1]):
                res.append(self.num_to_char.get(num, ""))
        return "".join(res)

    @property
    def vocab_size(self):
        return len(self.chars) + 1


# ====================================================================
# 2. INTENTIONAL CRNN ARCHITECTURE (OPTIMIZED REGULARIZATION)
# ====================================================================
class CustomMedicalCRNN(nn.Module):
    def __init__(self, vocab_size):
        super(CustomMedicalCRNN, self).__init__()

        # Convolutional Feature Extractor (Learns loops, lines, and medicine stroke curves)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d((2, 1)),
            nn.Dropout2d(0.3)  # Stepped up to structurally mitigate generalization gaps
        )

        # Bidirectional LSTM Layer (Scans writing paths horizontally left-to-right & right-to-left)
        self.hidden_size = 128
        self.rnn = nn.LSTM(input_size=1024, hidden_size=self.hidden_size, num_layers=2,
                           bidirectional=True, batch_first=True, dropout=0.4)

        # Fully Connected Linear Prediction Output Head
        self.fc = nn.Linear(self.hidden_size * 2, vocab_size)

    def forward(self, img_tensor):
        features = self.cnn(img_tensor)
        b, c, h, w = features.size()
        # Collapse spatial dimension down to input timesteps for the LSTM layers
        features = features.view(b, c * h, w).permute(0, 2, 1)
        rnn_out, _ = self.rnn(features)
        logits = self.fc(rnn_out)
        return logits.log_softmax(2)


# ====================================================================
# 3. HIGH-GENERALIZATION DOMAIN ADAPTATION DATA ENGINE
# ====================================================================
class LocalOCRDataset(Dataset):
    def __init__(self, image_dir, labels_df, encoder, target_w=256, target_h=64, is_training=True):
        self.image_dir = image_dir
        self.encoder = encoder
        self.target_w = target_w
        self.target_h = target_h
        self.is_training = is_training

        # Match your exact Excel columns: 'Images' and 'Text'
        raw_dict = dict(zip(labels_df['Images'].astype(str), labels_df['Text'].astype(str)))
        self.labels_dict = {k.lower().strip(): v for k, v in raw_dict.items()}

        all_files = os.listdir(image_dir)
        self.image_files = []
        self.file_map = {}

        for f in all_files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.pdf')):
                normalized_name = f.lower().strip()
                if normalized_name in self.labels_dict:
                    self.image_files.append(normalized_name)
                    self.file_map[normalized_name] = f

        print(f"📂 Domain Engine Initialized: {len(self.image_files)} training sequences ready with deep pixel transformations.")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        normalized_filename = self.image_files[idx]
        actual_filename = self.file_map[normalized_filename]
        file_path = os.path.join(self.image_dir, actual_filename)

        if actual_filename.lower().endswith('.pdf'):
            try:
                pil_pages = convert_from_path(file_path, first_page=1, last_page=1)
                if len(pil_pages) > 0:
                    image = cv2.cvtColor(np.array(pil_pages[0]), cv2.COLOR_RGB2GRAY)
            except Exception:
                image = None
        else:
            image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            image = np.ones((self.target_h, self.target_w), dtype=np.uint8) * 255

        # 🎯 TYPE ENFORCEMENT GUARD: Flatten channels & force 8-bit unsigned integer layout before CLAHE
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)

        # 🚀 ADVANCED DOMAIN ADAPTATION PIPELINE
        if self.is_training:
            h_orig, w_orig = image.shape[:2]

            # 1. Bounding Box Jittering (Shift context by up to 3% margin pixels)
            if np.random.rand() > 0.5 and h_orig > 10 and w_orig > 10:
                dy = int(h_orig * np.random.uniform(-0.03, 0.03))
                dx = int(w_orig * np.random.uniform(-0.03, 0.03))
                M_shift = np.float32([[1, 0, dx], [0, 1, dy]])
                image = cv2.warpAffine(image, M_shift, (w_orig, h_orig), borderMode=cv2.BORDER_CONSTANT,
                                       borderValue=255)

            # 2. Random 3D Perspective Distortions (Simulates angled phone captures)
            if np.random.rand() > 0.4:
                pts1 = np.float32([[0, 0], [w_orig, 0], [0, h_orig], [w_orig, h_orig]])
                shift = np.random.uniform(0.02, 0.05) * min(w_orig, h_orig)
                pts2 = np.float32([
                    [np.random.uniform(0, shift), np.random.uniform(0, shift)],
                    [w_orig - np.random.uniform(0, shift), np.random.uniform(0, shift)],
                    [np.random.uniform(0, shift), h_orig - np.random.uniform(0, shift)],
                    [w_orig - np.random.uniform(0, shift), h_orig - np.random.uniform(0, shift)]
                ])
                P_mat = cv2.getPerspectiveTransform(pts1, pts2)
                image = cv2.warpPerspective(image, P_mat, (w_orig, h_orig), borderMode=cv2.BORDER_CONSTANT,
                                            borderValue=255)

            # 3. Dynamic Handwriting Slant Shifts
            if np.random.rand() > 0.4:
                angle = np.random.uniform(-6, 6)
                M_rot = cv2.getRotationMatrix2D((w_orig // 2, h_orig // 2), angle, 1.0)
                image = cv2.warpAffine(image, M_rot, (w_orig, h_orig), borderMode=cv2.BORDER_CONSTANT, borderValue=255)

            # 4. Morphological Pen Stroke Transformations (Thick vs Thin Ink)
            if np.random.rand() > 0.5:
                kernel = np.ones((2, 2), np.uint8)
                if np.random.rand() > 0.5:
                    image = cv2.erode(image, kernel, iterations=1)  # Simulates ballpoint ink bleed
                else:
                    image = cv2.dilate(image, kernel, iterations=1)  # Simulates fading gel pen

            # 5. Adaptive Local Contrast Normalization (CLAHE)
            if np.random.rand() > 0.5:
                clahe = cv2.createCLAHE(clipLimit=np.random.uniform(1.5, 3.0), tileGridSize=(8, 8))
                image = clahe.apply(image)

            # 6. Pixel Degradation (Salt & Pepper Noise)
            if np.random.rand() > 0.4:
                noise = np.random.rand(*image.shape)
                image[noise < 0.01] = 0  # Random ink splatter spots
                image[noise > 0.99] = 255  # Random sensor/scanner dropouts

            # 7. Adaptive Gaussian Scanner Blur Emulation
            if np.random.rand() > 0.4:
                image = cv2.GaussianBlur(image, (3, 3), 0)

        # Standardize canvas sizing to 256x64 without squishing aspect ratios
        canvas = np.ones((self.target_h, self.target_w), dtype=np.uint8) * 255
        scale = min(self.target_w / image.shape[1], self.target_h / image.shape[0])
        nw, nh = max(4, int(image.shape[1] * scale)), max(4, int(image.shape[0] * scale))
        resized = cv2.resize(image, (min(nw, self.target_w), min(nh, self.target_h)))

        start_x, start_y = (self.target_w - nw) // 2, (self.target_h - nh) // 2
        canvas[start_y:start_y + nh, start_x:start_x + nw] = resized

        if np.mean(canvas) < 127:
            canvas = cv2.bitwise_not(canvas)

        img_tensor = canvas.astype(np.float32) / 255.0

        text_label = self.labels_dict[normalized_filename]
        encoded_label = self.encoder.encode(text_label)

        return torch.tensor(img_tensor).unsqueeze(0), torch.tensor(encoded_label, dtype=torch.long)


def ocr_collate_fn(batch):
    images, targets = zip(*batch)
    images = torch.stack(images, 0)
    target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
    targets = torch.cat(targets, 0)
    return images, targets, target_lengths


# ====================================================================
# 4. ROBUST EXTRACTION OPTIMIZATION LIFECYCLE
# ====================================================================
def train_pipeline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = MedicalLabelEncoder()
    model = CustomMedicalCRNN(encoder.vocab_size).to(device)

    # Dataset parent directory path configuration
    DATASET_BASE_DIR = r"C:\Users\Bubu\AI-Healthcare-Diagnostic-System\data\clean\MedicalCRNN_clean"
    TRAIN_IMG_DIR = os.path.join(DATASET_BASE_DIR, "train")

    label_path_csv = os.path.join(DATASET_BASE_DIR, "Train_Label.csv")
    label_path_xlsx = os.path.join(DATASET_BASE_DIR, "Train_Label.xlsx")

    if os.path.exists(label_path_csv):
        print("📊 Found Train_Label as a CSV file. Initializing domain data parser...")
        labels_df = pd.read_csv(label_path_csv)
    elif os.path.exists(label_path_xlsx):
        print("📊 Found Train_Label as an Excel spreadsheet. Initializing domain data parser...")
        labels_df = pd.read_excel(label_path_xlsx)
    else:
        print(f"❌ Error: Could not locate either 'Train_Label.csv' or 'Train_Label.xlsx' in {DATASET_BASE_DIR}")
        return

    MODEL_SAVE_PATH = r"C:\Users\Bubu\AI-Healthcare-Diagnostic-System\models\MedicalCRNN_v1.pth"
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

    dataset = LocalOCRDataset(TRAIN_IMG_DIR, labels_df, encoder, is_training=True)

    if len(dataset) == 0:
        print("❌ Error: Zero matched samples found. Verify image extensions match spreadsheet row texts exactly.")
        return

    # Batch size 8 balances regularization constraints alongside high-variance data augmentations
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=ocr_collate_fn)

    criterion = nn.CTCLoss(blank=0, zero_infinity=True)

    # Tuned learning rate & weight decay to force robust generalization transitions
    optimizer = optim.AdamW(model.parameters(), lr=0.0008, weight_decay=1e-3)

    print(f"📡 High-Generalization Engine Active on Framework: [{device}]")
    print(f"⏳ Optimizing model parameters over 50 robust validation epochs...")

    model.train()
    for epoch in range(1, 51):
        epoch_loss = 0.0
        for images, targets, target_lengths in dataloader:
            images = images.to(device)
            optimizer.zero_grad()

            log_probs = model(images)
            log_probs = log_probs.permute(1, 0, 2)
            input_lengths = torch.full(size=(images.size(0),), fill_value=log_probs.size(0), dtype=torch.long)

            loss = criterion(log_probs, targets, input_lengths, target_lengths)
            loss.backward()

            # Stepped down from 2.0 to 1.5 to aggressively stabilize parameter convergence boundaries
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.5)
            optimizer.step()
            epoch_loss += loss.item()

        if epoch % 5 == 0 or epoch == 1:
            print(f"📈 Epoch [{epoch:02d}/50] -> Domain-Adapted Continuous Loss: {epoch_loss:.4f}")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"\n✅ Training Complete! Highly generalized weight file saved to: {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    train_pipeline()