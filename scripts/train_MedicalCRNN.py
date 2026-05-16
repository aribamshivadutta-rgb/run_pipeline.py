import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from PIL import Image
import pandas as pd
import torchvision.transforms as T
from tqdm import tqdm
import os


# --- 1. ENCODER ---
# Maps characters to numbers for CTC Loss.
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


# --- 2. DATASET LOADER ---
class MedicalDataset(Dataset):
    def __init__(self, csv_file, img_dir, encoder):
        # Using ISO-8859-1 to handle special characters in medical names
        try:
            self.df = pd.read_csv(csv_file, encoding='utf-8')
        except:
            self.df = pd.read_csv(csv_file, encoding='ISO-8859-1')

        self.img_dir = img_dir
        self.encoder = encoder
        self.transform = T.Compose([
            T.Grayscale(),
            T.Resize((64, 256)),
            T.ToTensor(),
            T.Normalize((0.5,), (0.5,))
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx, 0]
        label_text = self.df.iloc[idx, 1]
        img_path = os.path.join(self.img_dir, str(img_name))

        if not os.path.exists(img_path):
            return torch.zeros((1, 64, 256)), torch.LongTensor([0]), torch.LongTensor([1])

        image = Image.open(img_path).convert("L")
        image = self.transform(image)
        label = torch.LongTensor(self.encoder.encode(label_text))
        return image, label, torch.LongTensor([len(label)])


# --- 3. ARCHITECTURE: MedicalCRNN ---
class MedicalCRNN(nn.Module):
    def __init__(self, vocab_size):
        super(MedicalCRNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(), nn.MaxPool2d((2, 1))
        )
        self.rnn = nn.LSTM(2048, 256, bidirectional=True, num_layers=2, batch_first=True)
        self.fc = nn.Linear(512, vocab_size)

    def forward(self, x):
        x = self.cnn(x)
        b, c, h, w = x.size()
        x = x.view(b, w, c * h)
        x, _ = self.rnn(x)
        x = self.fc(x)
        return x.log_softmax(2)


# --- 4. TRAINING EXECUTION ---
def run_training():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = MedicalLabelEncoder()

    # --- DYNAMIC PATH DETECTION ---
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))

    # Path to the CLEAN data created by your preprocessor
    CLEAN_BASE = os.path.join(PROJECT_ROOT, 'data', 'clean', 'MedicalCRNN_clean')
    TRAIN_CSV = os.path.join(CLEAN_BASE, 'Train_Label.csv')
    TRAIN_DIR = os.path.join(CLEAN_BASE, 'train')

    # Path to save the trained model
    MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, 'models', 'MedicalCRNN_v1.pth')
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

    if not os.path.exists(TRAIN_CSV):
        print(f"ERROR: Cannot find labels at {TRAIN_CSV}. Run preprocessor first.")
        return

    # Load Data
    train_ds = MedicalDataset(TRAIN_CSV, TRAIN_DIR, encoder)

    def collate_fn(batch):
        images, labels, lengths = zip(*batch)
        return torch.stack(images, 0), \
            nn.utils.rnn.pad_sequence(labels, batch_first=True), \
            torch.cat(lengths, 0)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_fn)

    model = MedicalCRNN(encoder.vocab_size).to(device)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    print(f"Starting Training on {len(train_ds)} images...")
    print(f"Model will be saved to: {MODEL_SAVE_PATH}")

    for epoch in range(50):
        model.train()
        epoch_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")

        for imgs, labels, lengths in progress_bar:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()

            preds = model(imgs).permute(1, 0, 2)
            input_lens = torch.full((imgs.size(0),), preds.size(0), dtype=torch.long)

            loss = criterion(preds, labels, input_lens, lengths)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        print(f"Epoch {epoch + 1} Complete. Avg Loss: {epoch_loss / len(train_loader):.4f}")
        torch.save(model.state_dict(), MODEL_SAVE_PATH)


if __name__ == "__main__":
    run_training()