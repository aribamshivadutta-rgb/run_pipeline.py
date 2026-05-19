import os
import sys
import torch
import torch.nn as nn
import cv2
import numpy as np

# Ensure project root is in the system path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)


# ====================================================================
# 1. MATCHING ENCODER & ARCHITECTURE DEFINITIONS
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

    # 🚀 FIX: Add the missing property decorator so PyTorch can read the vocab size integer!
    @property
    def vocab_size(self):
        return len(self.chars) + 1


class MedicalCRNN(nn.Module):
    def __init__(self, vocab_size):
        super(MedicalCRNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d((2, 1)),
            nn.Dropout2d(0.3)
        )
        self.hidden_size = 256
        self.num_layers = 2
        self.rnn = nn.LSTM(input_size=2048, hidden_size=self.hidden_size, num_layers=self.num_layers,
                           bidirectional=True, batch_first=True, dropout=0.4)
        self.fc = nn.Linear(self.hidden_size * 2, vocab_size)

    def forward(self, img_tensor):
        features = self.cnn(img_tensor)
        b, c, h, w = features.size()
        features = features.view(b, c * h, w).permute(0, 2, 1)
        rnn_out, _ = self.rnn(features)
        logits = self.fc(rnn_out)
        return logits.log_softmax(2)


# ====================================================================
# 2. ISOLATED INFERENCE EXECUTION ENGINE
# ====================================================================
def test_single_image(image_filename, model_weights_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = MedicalLabelEncoder()

    # 1. Initialize and Load Model
    model = MedicalCRNN(encoder.vocab_size).to(device)
    if not os.path.exists(model_weights_path):
        print(f"❌ Error: Weights file not found at {model_weights_path}")
        return

    raw_state_dict = torch.load(model_weights_path, map_location=device)
    sanitized_state_dict = {k.replace("module.", ""): v for k, v in raw_state_dict.items()}
    model.load_state_dict(sanitized_state_dict, strict=True)
    model.eval()

    # 2. Locate and Read Target Image
    possible_paths = [
        image_filename,
        os.path.join(PROJECT_ROOT, image_filename),
        os.path.join(PROJECT_ROOT, "data", "temp_samples", image_filename),
        os.path.join(PROJECT_ROOT, "data", "raw", image_filename)
    ]

    image = None
    actual_path = ""
    for path in possible_paths:
        if os.path.exists(path) and not os.path.isdir(path):
            image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            actual_path = path
            break

    if image is None:
        print(f"❌ Error: Target file '{image_filename}' could not be located in project workspace.")
        return

    print(f"\n📸 Processing Image Source: {actual_path}")
    print(f"📐 Original Image Geometry Dimensions: {image.shape[1]}x{image.shape[0]}")

    # 3. Process Tensors (Zero-Centered Normalization Matrix Alignment)
    target_w, target_h = 256, 64
    canvas = np.ones((target_h, target_w), dtype=np.uint8) * 255

    scale = min(target_w / image.shape[1], target_h / image.shape[0])
    nw, nh = max(4, int(image.shape[1] * scale)), max(4, int(crop_shape_h := image.shape[0] * scale))
    resized = cv2.resize(image, (min(nw, target_w), min(nh, target_h)))

    start_x = (target_w - nw) // 2
    start_y = (target_h - nh) // 2
    canvas[start_y:start_y + nh, start_x:start_x + nw] = resized

    if np.mean(canvas) < 127:
        canvas = cv2.bitwise_not(canvas)

    normalized_input = canvas.astype(np.float32) / 255.0
    normalized_input = (normalized_input - 0.5) / 0.5  # Zero-centered alignment pass

    tensor_input = torch.from_numpy(normalized_input).float().to(device).unsqueeze(0).unsqueeze(0)

    # 4. Forward Propagation & Decoding Pass
    with torch.no_grad():
        logits = model(tensor_input)
        probs = torch.exp(logits).squeeze(0)
        best_path = torch.argmax(logits.squeeze(0), dim=1).cpu().numpy()

        path_probs = probs[torch.arange(probs.size(0)), best_path].cpu().numpy()
        confidence = float(np.mean(path_probs)) * 100
        active_indices = [int(idx) for idx in best_path if idx != 0]
        decoded_string = encoder.decode(best_path).strip()

    print(f"🌡️ Mean Log Probability Sequence Confidence: {confidence:.2f}%")
    print(f"🔢 Raw Extracted Argmax Path Token Arrays:\n{list(best_path)}")
    print(f"🎯 Non-Zero Active Vocabulary Character Map Indices:\n{active_indices}")
    print(f"🔮 Direct Model Output Text: \"{decoded_string}\"")


if __name__ == "__main__":
    # Test file parameters
    target_weights = os.path.join(PROJECT_ROOT, "models", "MedicalCRNN_v1.pth")

    # Run test on P0004.jpg
    test_single_image("blur.jpg", target_weights)