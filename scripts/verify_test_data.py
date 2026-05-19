import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import pandas as pd
import os


# ====================================================================
# 1. ENCODER ALIGNMENT
# ====================================================================
class MedicalLabelEncoder:
    def __init__(self):
        self.chars = " %()-./012345678?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        self.char_to_num = {char: i + 1 for i, char in enumerate(self.chars)}
        self.num_to_char = {i + 1: char for i, char in enumerate(self.chars)}

    def encode(self, text):
        return [self.char_to_num[c] for c in str(text) if c in self.char_to_num]

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
# 2. FIXED DATASET LOADER (Deterministic Evaluation Mode)
# ====================================================================
class MedicalDataset(Dataset):
    def __init__(self, csv_file, img_dir, encoder):
        try:
            self.df = pd.read_csv(csv_file, encoding='utf-8')
        except:
            self.df = pd.read_csv(csv_file, encoding='ISO-8859-1')
        self.img_dir = img_dir
        self.encoder = encoder

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx, 0]
        label_text = self.df.iloc[idx, 1]
        img_path = os.path.join(self.img_dir, str(img_name))

        if not os.path.exists(img_path) or pd.isna(label_text):
            return torch.zeros((1, 64, 256)), torch.LongTensor([0])

        raw_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if raw_img is None:
            return torch.zeros((1, 64, 256)), torch.LongTensor([0])

        # Binarization Profile
        if np.mean(raw_img) > 127:
            _, processed_img = cv2.threshold(raw_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            _, processed_img = cv2.threshold(raw_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Smart Crop: Extract text boundary coordinates and isolate ink payload
        pts = np.argwhere(processed_img == 0)
        if len(pts) > 0:
            y_min, x_min = pts.min(axis=0)
            y_max, x_max = pts.max(axis=0)
            processed_img = processed_img[y_min:y_max + 1, x_min:x_max + 1]

        target_w, target_h = 256, 64
        padded_canvas = np.ones((target_h, target_w), dtype=np.uint8) * 255

        # Aspect Ratio Preservation Engine
        h_img, w_img = processed_img.shape
        scale = target_h / h_img
        nw = int(w_img * scale)
        nh = target_h

        if nw > target_w:
            scale = target_w / w_img
            nw = target_w
            nh = int(h_img * scale)

        resized_crop = cv2.resize(processed_img, (max(4, nw), max(4, nh)))
        start_x = max(0, (target_w - nw) // 2)
        start_y = max(0, (target_h - nh) // 2)
        padded_canvas[start_y:start_y + nh, start_x:start_x + nw] = resized_crop

        # High-Contrast Tensor Mapping (-1.0 to 1.0 scaling matrix)
        tensor_img = padded_canvas.astype(np.float32) / 255.0
        tensor_img = (tensor_img - 0.5) / 0.5
        tensor_img = torch.from_numpy(tensor_img).unsqueeze(0).float()

        label_encoded = self.encoder.encode(label_text)
        if not label_encoded:
            label_encoded = [0]

        return tensor_img, torch.LongTensor(label_encoded)


# ====================================================================
# 3. CORRECTED ARCHITECTURE BLOCK (Fully Synchronized)
# ====================================================================
class MedicalCRNN(nn.Module):
    def __init__(self, vocab_size):
        super(MedicalCRNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d((2, 1))
        )
        # 🎯 ALIGNED CAPACITY: Upgraded to 256 to sync with your high-generalization model
        self.hidden_size = 256
        self.num_layers = 2
        self.rnn = nn.LSTM(input_size=1024, hidden_size=self.hidden_size, num_layers=self.num_layers,
                           bidirectional=True, batch_first=True)
        self.fc = nn.Linear(self.hidden_size * 2, vocab_size)

    def forward(self, img_tensor, hx=None):
        features = self.cnn(img_tensor)
        b, c, h, w = features.size()
        features = features.view(b, c * h, w).permute(0, 2, 1)
        rnn_out, _ = self.rnn(features, hx)
        logits = self.fc(rnn_out)
        return logits.log_softmax(2)


# ====================================================================
# 4. BATCHED PERFORMANCE VERIFICATION PIPELINE (TEST DATASET TARGET)
# ====================================================================
def verify_on_test_data():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = MedicalLabelEncoder()

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
    CLEAN_BASE = os.path.join(PROJECT_ROOT, 'data', 'clean', 'MedicalCRNN_clean')

    # 🎯 TARGET ALIGNMENT: Pointing explicitly to unseen test parameters
    TEST_CSV = os.path.join(CLEAN_BASE, 'Test_Label.csv')
    TEST_DIR = os.path.join(CLEAN_BASE, 'test')

    MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, 'models', 'MedicalCRNN_v1.pth')

    if not os.path.exists(MODEL_SAVE_PATH):
        print(f"❌ ERROR: Trained weight file not found at: {MODEL_SAVE_PATH}")
        return

    eval_ds = MedicalDataset(TEST_CSV, TEST_DIR, encoder)

    def collate_fn(batch):
        images, labels = zip(*batch)
        return torch.stack(images, 0), nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=0)

    eval_loader = DataLoader(eval_ds, batch_size=32, shuffle=False, collate_fn=collate_fn)

    print(f"🔄 Initializing Upgraded Model Architecture & Loading Weights from: {MODEL_SAVE_PATH}")
    model = MedicalCRNN(encoder.vocab_size).to(device)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    model.eval()

    correct_predictions = 0
    total_samples = 0
    sample_print_counter = 0

    print(f"\n🔍 Running Model Recognition on {len(eval_ds)} UNSEEN Test Images:\n" + "=" * 50)

    with torch.no_grad():
        for imgs, labels in eval_loader:
            imgs = imgs.to(device)
            batch_size = imgs.size(0)

            num_directions = 2
            h0 = torch.zeros(model.num_layers * num_directions, batch_size, model.hidden_size).to(device)
            c0 = torch.zeros(model.num_layers * num_directions, batch_size, model.hidden_size).to(device)

            preds = model(imgs, (h0, c0)).permute(1, 0, 2)

            for b_idx in range(batch_size):
                if torch.sum(imgs[b_idx]) == 0:
                    continue

                single_pred_path = torch.argmax(preds[:, b_idx, :], dim=1).cpu().numpy()
                predicted_text = encoder.decode(single_pred_path).strip()

                single_label = labels[b_idx].cpu().numpy()
                true_text = encoder.decode(single_label).strip()

                is_match = (predicted_text == true_text)
                if is_match:
                    correct_predictions += 1
                total_samples += 1
                sample_print_counter += 1

                # Show visual confirmation logs for accuracy verification tracing
                if sample_print_counter <= 50 or true_text.lower() == "rivotril":
                    status_icon = "✅" if is_match else "❌"
                    print(f"Sample #{sample_print_counter:03d} {status_icon}")
                    print(f"  ├─ Ground Truth: '{true_text}'")
                    print(f"  └─ Predicted:    '{predicted_text}'")
                    print("-" * 40)

    accuracy = (correct_predictions / total_samples) * 100 if total_samples > 0 else 0
    print("=" * 50)
    print(f"📊 TEST DATASET SUMMARY (UPGRADED CAPACITY):")
    print(f"   🎯 Exact Match Accuracy: {accuracy:.2f}% ({correct_predictions}/{total_samples})")


if __name__ == "__main__":
    verify_on_test_data()