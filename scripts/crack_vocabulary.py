import torch
import os
import sys

CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_SCRIPT_DIR)
WEIGHTS_PATH = r"C:\Users\Bubu\AI-Healthcare-Diagnostic-System\models\MedicalCRNN_v1.pth"

# Emulate the exact layer shapes found in your file
import torch.nn as nn


class MedicalCRNN(nn.Module):
    def __init__(self, vocab_size=78):  # Standard dataset vocabulary base sizing
        super(MedicalCRNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(), nn.MaxPool2d((2, 1))
        )
        self.rnn = nn.LSTM(input_size=2048, hidden_size=256, num_layers=2, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(512, vocab_size)


# Load weights and extract the output vocabulary dimension
state_dict = torch.load(WEIGHTS_PATH, map_location="cpu")
fc_weight_shape = state_dict["fc.weight"].shape
trained_vocab_size = fc_weight_shape[0]

print("================ VOCABULARY DEEP EXPLORER ================")
print(f"🎯 Model Output Classification Layers: {trained_vocab_size} distinct characters.")

# Common standardized vocabulary string variations used during CRNN training sets
variations = [
    # Standard alphanumeric sequence with shifting padding maps
    "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-.",
    "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ %()-./012345678?",
    " %()-./012345678?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
]

print("\n📋 Checking Target Mapping Offsets against Trained Indices:")
for v_str in variations:
    actual_len = len(v_str) + 1  # Include CTC blank token offset
    if actual_len == trained_vocab_size:
        print(f"  🟢 MATCH FOUND! Your model maps to an alphabet string of length {len(v_str)}:")
        print(f"  ➡️ String to use: \"{v_str}\"")
        exit()

print("  ❌ No standard string length matches the model's exact shape of fields.")
print(f"  Fallback: Model demands a string containing exactly {trained_vocab_size - 1} characters.")