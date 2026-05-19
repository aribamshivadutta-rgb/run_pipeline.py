import torch
import os

CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_SCRIPT_DIR)
WEIGHTS_PATH = r"C:\Users\Bubu\AI-Healthcare-Diagnostic-System\models\MedicalCRNN_v1.pth"

if not os.path.exists(WEIGHTS_PATH):
    print(f"❌ Cannot locate weights file at: {WEIGHTS_PATH}")
    exit()

print("🔍 Opening weights archive file structure...")
try:
    state_dict = torch.load(WEIGHTS_PATH, map_location="cpu")
    print(f"✅ Loaded successfully! Total dictionary weight parameters found: {len(state_dict)}")
    print("\n📋 Param Tracking Layer Layer Keys:")

    cnn_keys = 0
    rnn_keys = 0

    for key in list(state_dict.keys())[:15]:
        print(f"  ├─ {key} -> Tensor Shape: {list(state_dict[key].shape)}")
        if "cnn" in key: cnn_keys += 1
        if "rnn" in key: rnn_keys += 1

    print("  └─ ... [Truncated list]")

    # Mathematical validation check: Are weights randomized or actual trained values?
    first_layer_key = list(state_dict.keys())[0]
    sample_tensor = state_dict[first_layer_key].numpy()

    print("\n🔢 Structural Values Check:")
    print(f"  ├─ Layer Evaluated: {first_layer_key}")
    print(f"  ├─ Weights Mean Value Variance: {sample_tensor.mean():.6f}")
    print(f"  └─ Weights Standard Deviation Profile: {sample_tensor.std():.6f}")

    if abs(sample_tensor.mean()) < 1e-7 and abs(sample_tensor.std() - 1.0) < 1e-7:
        print("\n🚨 DIAGNOSIS: These weights look like a generic uninitialized identity matrix!")
    else:
        print(
            "\n🎯 DIAGNOSIS: Physical values exist in this file, meaning the layers match a completely different class design.")

except Exception as err:
    print(f"❌ Failed to parse weight parameters: {err}")