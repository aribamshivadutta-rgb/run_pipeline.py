import os
import sys
import torch
import torch.nn as nn  # ◄── Added direct structural module binding link here
import cv2
import numpy as np
import pandas as pd

CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_SCRIPT_DIR)
sys.path.append(PROJECT_ROOT)


# ====================================================================
# 1. IDENTICAL LAYER ARCHITECTURE FOR TESTING MATCHES
# ====================================================================
class MedicalLabelEncoder:
    def __init__(self):
        self.chars = " %()-./012345678?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        self.char_to_num = {char: i + 1 for i, char in enumerate(self.chars)}
        self.num_to_char = {i + 1: char for i, char in enumerate(self.chars)}

    def decode(self, nums):
        res = []
        for i, num in enumerate(nums):
            if num != 0 and (i == 0 or num != nums[i - 1]):
                res.append(self.num_to_char.get(num, ""))
        return "".join(res)

    @property
    def vocab_size(self):
        return len(self.chars) + 1


class CustomMedicalCRNN(nn.Module):
    def __init__(self, vocab_size):
        super(CustomMedicalCRNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d((2, 1)),
            nn.Dropout2d(0.2)
        )
        self.hidden_size = 128
        self.rnn = nn.LSTM(input_size=1024, hidden_size=self.hidden_size, num_layers=2,
                           bidirectional=True, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(self.hidden_size * 2, vocab_size)

    def forward(self, img_tensor):
        features = self.cnn(img_tensor)
        b, c, h, w = features.size()
        features = features.view(b, c * h, w).permute(0, 2, 1)
        rnn_out, _ = self.rnn(features)
        logits = self.fc(rnn_out)
        return logits.log_softmax(2)


# ====================================================================
# 2. RUNTIME INFERENCE ROUTINE
# ====================================================================
def evaluate_on_test_set():
    import torch.nn as nn  # Inline patch safety
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = MedicalLabelEncoder()
    model = CustomMedicalCRNN(encoder.vocab_size).to(device)

    # Define exact directories inside your Clean folder structure
    DATASET_BASE_DIR = r"C:\Users\Bubu\AI-Healthcare-Diagnostic-System\data\clean\MedicalCRNN_clean"
    TEST_IMG_DIR = os.path.join(DATASET_BASE_DIR, "test")
    WEIGHTS_PATH = r"C:\Users\Bubu\AI-Healthcare-Diagnostic-System\models\MedicalCRNN_v1.pth"

    # 1. Load your compiled weights file
    if not os.path.exists(WEIGHTS_PATH):
        print(f"❌ Error: Cannot find trained weights file at {WEIGHTS_PATH}. Let your training finish first!")
        return

    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    model.eval()
    print("🟢 Custom Scratch-Trained Weights loaded successfully. Processing test labels...")

    # 2. Open Test Sheet
    label_path_csv = os.path.join(DATASET_BASE_DIR, "Test_Label.csv")
    label_path_xlsx = os.path.join(DATASET_BASE_DIR, "Test_Label.xlsx")

    if os.path.exists(label_path_csv):
        test_df = pd.read_csv(label_path_csv)
    elif os.path.exists(label_path_xlsx):
        test_df = pd.read_excel(label_path_xlsx)
    else:
        print("❌ Error: Could not locate Test_Label sheet inside your directory folder.")
        return

    raw_dict = dict(zip(test_df['Images'].astype(str), test_df['Text'].astype(str)))

    # Scan the test folder
    test_files = [f for f in os.listdir(TEST_IMG_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    print(f"🔬 Found {len(test_files)} testing files. Running inference on sample entries:\n")
    print(f"{'IMAGE FILE':<15} | {'EXPECTED (TRUE LABEL)':<25} | {'MODEL PREDICTION (RAW OUTPUT)'}")
    print("-" * 75)

    # Test up to the first 15 files to inspect the text outputs clearly
    sample_limit = min(15, len(test_files))
    matched_count = 0

    for idx in range(sample_limit):
        filename = test_files[idx]
        true_text = raw_dict.get(filename, "Unmapped")

        # Preprocess exactly like the dataloader geometry
        img_path = os.path.join(TEST_IMG_DIR, filename)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            continue

        target_w, target_h = 256, 64
        canvas = np.ones((target_h, target_w), dtype=np.uint8) * 255
        scale = min(target_w / image.shape[1], target_h / image.shape[0])
        nw, nh = max(4, int(image.shape[1] * scale)), max(4, int(image.shape[0] * scale))
        resized = cv2.resize(image, (min(nw, target_w), min(nh, target_h)))

        start_x, start_y = (target_w - nw) // 2, (target_h - nh) // 2
        canvas[start_y:start_y + nh, start_x:start_x + nw] = resized
        if np.mean(canvas) < 127:
            canvas = cv2.bitwise_not(canvas)

        img_tensor = canvas.astype(np.float32) / 255.0
        tensor_input = torch.tensor(img_tensor).unsqueeze(0).unsqueeze(0).to(device)

        # Forward Pass evaluation
        with torch.no_grad():
            log_probs = model(tensor_input)
            preds = log_probs.argmax(dim=2).squeeze(0).cpu().numpy()

        predicted_text = encoder.decode(preds).strip()
        print(f"{filename:<15} | {true_text:<25} | {predicted_text}")


if __name__ == "__main__":
    evaluate_on_test_set()