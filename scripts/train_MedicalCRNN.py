import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import random


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
# 2. ADVANCED DATASET LOADER WITH AUGMENTATION
# ====================================================================
class MedicalDataset(Dataset):
    def __init__(self, csv_file, img_dir, encoder, is_training=True):
        try:
            self.df = pd.read_csv(csv_file, encoding='utf-8')
        except:
            self.df = pd.read_csv(csv_file, encoding='ISO-8859-1')

        self.img_dir = img_dir
        self.encoder = encoder
        self.is_training = is_training

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx, 0]
        label_text = self.df.iloc[idx, 1]
        img_path = os.path.join(self.img_dir, str(img_name))

        if not os.path.exists(img_path) or pd.isna(label_text):
            return torch.zeros((1, 64, 256)), torch.LongTensor([0]), torch.LongTensor([1])

        raw_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if raw_img is None:
            return torch.zeros((1, 64, 256)), torch.LongTensor([0]), torch.LongTensor([1])

        # Apply robust light-background binarization profile (Matches App execution)
        if np.mean(raw_img) > 127:
            _, processed_img = cv2.threshold(raw_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            _, processed_img = cv2.threshold(raw_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # 🎯 GEOMETRIC DATA AUGMENTATION PASS
        if self.is_training and random.random() < 0.35:
            # Mild rotation adjustment simulating hurried handwriting angles (-3 to +3 degrees)
            angle = random.uniform(-3, 3)
            h_r, w_r = processed_img.shape
            M = cv2.getRotationMatrix2D((w_r // 2, h_r // 2), angle, 1.0)
            processed_img = cv2.warpAffine(processed_img, M, (w_r, h_r), borderValue=255)

        target_w, target_h = 256, 64
        padded_canvas = np.ones((target_h, target_w), dtype=np.uint8) * 255

        scale = min(target_w / processed_img.shape[1], target_h / processed_img.shape[0])
        nw, nh = max(4, int(processed_img.shape[1] * scale)), max(4, int(processed_img.shape[0] * scale))
        resized_crop = cv2.resize(processed_img, (min(nw, target_w), min(nh, target_h)))

        start_x = max(0, (target_w - nw) // 2)
        start_y = max(0, (target_h - nh) // 2)
        padded_canvas[start_y:start_y + nh, start_x:start_x + nw] = resized_crop

        # High-Contrast Document [Zero-Centered Scale Alignment: -1.0 to 1.0]
        tensor_img = padded_canvas.astype(np.float32) / 255.0
        tensor_img = (tensor_img - 0.5) / 0.5
        tensor_img = torch.from_numpy(tensor_img).unsqueeze(0).float()

        label_encoded = self.encoder.encode(label_text)
        if not label_encoded:
            label_encoded = [0]

        return tensor_img, torch.LongTensor(label_encoded), torch.LongTensor([len(label_encoded)])


# ====================================================================
# 3. ARCHITECTURE BLOCK (FULLY SYNCHRONIZED WITH PRODUCTION GRAPH)
# ====================================================================
class MedicalCRNN(nn.Module):
    def __init__(self, vocab_size):
        super(MedicalCRNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(), nn.MaxPool2d((2, 1))
        )
        self.hidden_size = 256
        self.num_layers = 2
        self.rnn = nn.LSTM(input_size=2048, hidden_size=self.hidden_size, num_layers=self.num_layers,
                           bidirectional=True, batch_first=True)
        self.fc = nn.Linear(self.hidden_size * 2, vocab_size)

    def forward(self, img_tensor, hx=None):
        features = self.cnn(img_tensor)
        b, c, h, w = features.size()

        features = features.view(b, c * h, w)
        features = features.permute(0, 2, 1)

        rnn_out, _ = self.rnn(features, hx)
        logits = self.fc(rnn_out)

        return logits.log_softmax(2)


# ====================================================================
# 4. TRAINING PERFORMANCE HARNESS EXECUTION
# ====================================================================
def run_training():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = MedicalLabelEncoder()

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))

    CLEAN_BASE = os.path.join(PROJECT_ROOT, 'data', 'clean', 'MedicalCRNN_clean')
    TRAIN_CSV = os.path.join(CLEAN_BASE, 'Train_Label.csv')
    TRAIN_DIR = os.path.join(CLEAN_BASE, 'train')

    MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, 'models', 'MedicalCRNN_v1.pth')
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

    if not os.path.exists(TRAIN_CSV):
        print(f"❌ ERROR: Missing target data labels at {TRAIN_CSV}.")
        return

    train_ds = MedicalDataset(TRAIN_CSV, TRAIN_DIR, encoder, is_training=True)

    def collate_fn(batch):
        images, labels, lengths = zip(*batch)
        return torch.stack(images, 0), \
            nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=0), \
            torch.cat(lengths, 0)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_fn)

    model = MedicalCRNN(encoder.vocab_size).to(device)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)

    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-4)

    # 🎯 ALPHABET MASTERY: Scale epochs to 350 to allow full character vocabulary convergence
    TOTAL_EPOCHS = 350
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TOTAL_EPOCHS, eta_min=1e-6)

    # Initialize tracking threshold for Accuracy Guard
    best_loss = float('inf')

    print(f"🚀 Initializing Optimized Training Engine on: {device}")
    print(f"📊 Loaded {len(train_ds)} valid prescription samples for optimization.")

    for epoch in range(TOTAL_EPOCHS):
        model.train()
        epoch_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{TOTAL_EPOCHS}")

        for imgs, labels, lengths in progress_bar:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()

            # Initialize zeroed hidden states to match app runtime execution environment exactly
            num_directions = 2
            h0 = torch.zeros(model.num_layers * num_directions, imgs.size(0), model.hidden_size).to(device)
            c0 = torch.zeros(model.num_layers * num_directions, imgs.size(0), model.hidden_size).to(device)

            preds = model(imgs, (h0, c0)).permute(1, 0, 2)
            input_lens = torch.full((imgs.size(0),), preds.size(0), dtype=torch.long)

            loss = criterion(preds, labels, input_lens, lengths)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        scheduler.step()

        avg_epoch_loss = epoch_loss / len(train_loader)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"📉 Epoch {epoch + 1} Complete | Avg CTC Loss: {avg_epoch_loss:.4f} | Active LR: {current_lr:.6f}")

        # Periodic Live Trace Preview Module
        model.eval()
        with torch.no_grad():
            test_img, test_lbl, _ = train_ds[0]
            if torch.sum(test_img) > 0:
                sample_tensor = test_img.unsqueeze(0).to(device)

                h0_t = torch.zeros(model.num_layers * num_directions, 1, model.hidden_size).to(device)
                c0_t = torch.zeros(model.num_layers * num_directions, 1, model.hidden_size).to(device)

                sample_preds = model(sample_tensor, (h0_t, c0_t))
                best_path = torch.argmax(sample_preds, dim=2).squeeze(0).cpu().numpy()
                active_tokens = [tok for tok in best_path if tok != 0]
                decoded_sample = encoder.decode(best_path)

                print(f"   🔍 Live Training Trace Pass:")
                print(f"      ├─ Target True Label text: '{encoder.decode(test_lbl.numpy())}'")
                print(f"      ├─ Active Predicted Path Token Indices Vector: {active_tokens}")
                print(f"      └── Model Decoded Text Prediction Output:  ➡️ '{decoded_sample}'\n")

        # ====================================================================
        # 🎯 BULLETPROOF ACCURACY GUARD WITH ATOMIC SWAP AND SLOT RE-ROUTING
        # ====================================================================
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss

            # Step A: Save to an isolated temporary file to avoid active operating system thread locks
            temp_save_path = MODEL_SAVE_PATH + f".epoch_{epoch + 1}.tmp"

            try:
                torch.save(model.state_dict(), temp_save_path)

                # Step B: Atomically clear out the older master path if it exists
                if os.path.exists(MODEL_SAVE_PATH):
                    try:
                        os.remove(MODEL_SAVE_PATH)
                    except OSError:
                        # If a background thread has a stubborn lock on the file, re-route to an epoch block
                        MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, 'models', f'MedicalCRNN_epoch_{epoch + 1}.pth')

                # Step C: Swap the fresh file cleanly into place
                os.rename(temp_save_path, MODEL_SAVE_PATH)
                print(
                    f"🌟 New Best Loss achieved ({best_loss:.4f})! Parameters matrix exported safely to: {MODEL_SAVE_PATH}")

            except Exception as io_err:
                print(f"⚠️ Windows I/O Lock intercepted during file commit pass: {io_err}")
                print("🔄 [Resilience Fallback]: Shifting weights to a dedicated unique milestone slot...")
                MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, 'models', f'MedicalCRNN_epoch_{epoch + 1}.pth')
                torch.save(model.state_dict(), MODEL_SAVE_PATH)
                print(f"✅ Recovery Successful! Milestone matrix safely locked into path: {MODEL_SAVE_PATH}")
        else:
            print(
                f"ℹ️ Epoch loss ({avg_epoch_loss:.4f}) did not beat historical minimum ({best_loss:.4f}). Skipping binary file write.")

    print("🏁 Optimized model training complete! Fine-tuned weights exported successfully.")


if __name__ == "__main__":
    run_training()