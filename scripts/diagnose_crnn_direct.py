import torch
import torch.nn as nn
import cv2
import numpy as np
import os
import sys

# Anchor paths relative to script location context
CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_SCRIPT_DIR)
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")


# ====================================================================
# 1. EMULATE PRODUCTION BACKBONE ARCHITECTURES
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


class MedicalCRNN(nn.Module):
    def __init__(self, vocab_size):
        super(MedicalCRNN, self).__init__()
        # CNN backbone matching weight file keys exactly: (0, 3, 6, 8)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(), nn.MaxPool2d((2, 1))
        )

        self.hidden_size = 256
        self.num_layers = 2

        # LSTM input matching weights configuration shape layout: [1024, 2048]
        self.rnn = nn.LSTM(
            input_size=2048,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bidirectional=True,
            batch_first=True
        )

        # Bidirectional network output map projection layers
        self.fc = nn.Linear(self.hidden_size * 2, vocab_size)

    def forward(self, img_tensor):
        features = self.cnn(img_tensor)
        b, c, h, w = features.size()

        # Spatial tensor permutation vectors
        features = features.view(b, c * h, w)
        features = features.permute(0, 2, 1)

        # 🎯 CHIP INITIALIZATION STATE GATES: Explicitly initialize recurrent layers to isolate memory leakage
        num_directions = 2  # Bidirectional
        h0 = torch.zeros(self.num_layers * num_directions, b, self.hidden_size).to(img_tensor.device)
        c0 = torch.zeros(self.num_layers * num_directions, b, self.hidden_size).to(img_tensor.device)

        rnn_out, _ = self.rnn(features, (h0, c0))
        logits = self.fc(rnn_out)
        return logits.log_softmax(2)


# ====================================================================
# 2. EXECUTIVE CORE DIRECT DIAGNOSTIC ENGINE
# ====================================================================
def execute_direct_diagnostic(image_path, weights_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = MedicalLabelEncoder()

    print(f"🧠 Loading CRNN Model Core onto: {device}...")
    model = MedicalCRNN(encoder.vocab_size).to(device)

    if not os.path.exists(weights_path):
        print(f"❌ Critical Failure: Weight file missing at {weights_path}")
        return

    # Sanitize module prefix headers out from the state dictionary layers cleanly
    raw_state_dict = torch.load(weights_path, map_location=device)
    sanitized_state_dict = {k.replace("module.", ""): v for k, v in raw_state_dict.items()}

    # Strictly bind weights across the unified multi-layer parameters matrix
    model.load_state_dict(sanitized_state_dict, strict=True)
    model.eval()
    print("✅ Model weights explicitly bound to network graph architecture!")

    print(f"📖 Unpacking Raw Target Image Matrix from: {image_path}...")
    raw_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if raw_img is None:
        print("❌ Critical Failure: Image path is unreadable, corrupt, or empty.")
        return

    h, w = raw_img.shape
    print(f"📐 Image Native Geometry Metrics: Height={h}px, Width={w}px")

    # High-aspect grid column quadrant parsing configurations
    slice_h = int(h * 0.08)
    slice_w = int(w * 0.50)
    stride_step = int(slice_h * 0.60)

    start_y = int(h * 0.10)
    end_y = int(h * 0.90)

    line_bounding_boxes = []
    # Generate multi-column strided quadrant trackers
    for start_x in [0, int(w * 0.50)]:
        for step_y in range(start_y, end_y - slice_h, stride_step):
            line_bounding_boxes.append((start_x, step_y, slice_w, slice_h))

    print(f"⚡ Emulated Slicer generated {len(line_bounding_boxes)} individual horizontal line frames.")
    print("----------------------------------------------------------------------\n")

    # Run direct tensor trace verification sequence on the first 2 segments
    for idx, (bx, by, bw, bh) in enumerate(line_bounding_boxes[:2]):
        line_crop = raw_img[by:by + bh, bx:bx + bw]

        if line_crop.size == 0 or line_crop.shape[0] < 2 or line_crop.shape[1] < 2:
            continue

        # Keep background LIGHT (255) and ink strokes DARK (0) for model compatibility
        if np.mean(line_crop) > 127:
            _, line_crop = cv2.threshold(line_crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            _, line_crop = cv2.threshold(line_crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        target_w, target_h = 256, 64
        # Pad canvas with pure white background (255) to avoid inverted gradient anomalies
        crnn_input_base = np.ones((target_h, target_w), dtype=np.uint8) * 255

        scale = min(target_w / line_crop.shape[1], target_h / line_crop.shape[0])
        nw, nh = max(4, int(line_crop.shape[1] * scale)), max(4, int(line_crop.shape[0] * scale))

        resized_crop = cv2.resize(line_crop, (min(nw, target_w), min(nh, target_h)))

        # Center text coordinates inside the target canvas bounds
        start_x = max(0, (target_w - nw) // 2)
        start_y = max(0, (target_h - nh) // 2)
        crnn_input_base[start_y:start_y + nh, start_x:start_x + nw] = resized_crop

        # Test using both normalization variants for execution tracing
        for tracking_mode in ["Zero-Centered ([-1, 1])", "Raw Intensity Matrix ([0, 1])"]:
            tensor_input = crnn_input_base.astype(np.float32) / 255.0

            if tracking_mode == "Zero-Centered ([-1, 1])":
                tensor_input = (tensor_input - 0.5) / 0.5

            img_tensor = torch.from_numpy(tensor_input).float().to(device).unsqueeze(0).unsqueeze(0)

            with torch.no_grad():
                preds = model(img_tensor)

                if list(preds.shape)[0] != 1 and list(preds.shape)[1] == 1:
                    preds = preds.permute(1, 0, 2)

                best_path = torch.argmax(preds, dim=2).squeeze(0).cpu().numpy()
                active_predictions = [token for token in best_path if token != 0]
                decoded_text = encoder.decode(best_path).strip()

            print(f"📊 [LINE DIAGNOSTIC TRACK RUN #{idx + 1}] -> MODE: {tracking_mode}")
            print(f"   ├─ Crop Pixel Dimensions: {line_crop.shape[1]}x{line_crop.shape[0]}")
            print(f"   ├─ Input Tensor Mean Value Variance: {np.mean(tensor_input):.4f}")
            print(f"   ├─ Raw Character Token Paths Array Vector:\n      {list(best_path)}")
            print(f"   ├─ Extracted Character Map Non-Zero Token Indices: {active_predictions}")
            print(f"   └── Decoded Text Output: ➡️ '{decoded_text}'\n")
        print("----------------------------------------------------------------------\n")


if __name__ == "__main__":
    # Point directly to your models directory weights file
    WEIGHTS = r"C:\Users\Bubu\AI-Healthcare-Diagnostic-System\models\MedicalCRNN_v1.pth"

    # TARGET TRACK: Set this directly to your target image file name inside your path
    TARGET_IMAGE = r"C:\Users\Bubu\AI-Healthcare-Diagnostic-System\blur.PNG"

    if os.path.exists(TARGET_IMAGE):
        print(f"✅ Found target image asset file. Initiating structural tensor tracing...")
        execute_direct_diagnostic(TARGET_IMAGE, WEIGHTS)
    else:
        FALLBACK_IMAGE = r"C:\Users\Bubu\AI-Healthcare-Diagnostic-System\sample_test.png"
        if os.path.exists(FALLBACK_IMAGE):
            print(f"✅ Found fallback image asset file. Initiating structural tensor tracing...")
            execute_direct_diagnostic(FALLBACK_IMAGE, WEIGHTS)
        else:
            print(f"❌ Critical Error: Could not locate a target test image inside your system path.")