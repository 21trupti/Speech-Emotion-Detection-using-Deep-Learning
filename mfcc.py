import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim

# -------------------- SETTINGS --------------------
AUDIO_DIR = r"C:\Users\ADMIN\Documents\Speech reg Project\audio"
SAMPLE_RATE = 16000
NUM_MFCC = 40
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001
# --------------------------------------------------

# -------------------- DATA LOADER --------------------
def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=NUM_MFCC)
        return np.mean(mfcc.T, axis=0)
    except Exception as e:
        print(f"❌ Feature extraction failed for {file_path}: {e}")
        return None

def load_data(folder_path):
    X = []
    y = []
    print(f"\n📂 Scanning subfolders in: {folder_path}")
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".wav"):
                path = os.path.join(root, file)
                try:
                    parts = file.split('-')
                    if len(parts) < 3:
                        print(f"❌ Skipping malformed filename: {file}")
                        continue
                    emotion_code = parts[2]
                    features = extract_features(path)
                    if features is None or features.shape[0] != NUM_MFCC:
                        continue
                    X.append(features)
                    y.append(emotion_code)
                except Exception as e:
                    print(f"❌ Error processing {file}: {e}")
    if len(X) == 0:
        raise RuntimeError("No valid audio files were loaded.")
    return np.array(X), np.array(y)
# --------------------------------------------------

# -------------------- DATASET CLASS --------------------
class SpeechDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
# --------------------------------------------------

# -------------------- MODEL --------------------
class EmotionClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(EmotionClassifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        return self.fc(x)
# --------------------------------------------------

# -------------------- MAIN --------------------
print("🔁 Loading data...")
X, y = load_data(AUDIO_DIR)
print(f"✅ Loaded {len(X)} samples.")

le = LabelEncoder()
y_encoded = le.fit_transform(y)
NUM_CLASSES = len(np.unique(y_encoded))

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

train_loader = DataLoader(SpeechDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(SpeechDataset(X_test, y_test), batch_size=BATCH_SIZE)

model = EmotionClassifier(input_dim=NUM_MFCC, num_classes=NUM_CLASSES)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print("🚀 Training model...")
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for features, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    avg_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.4f}")

# Evaluate
print("🧪 Evaluating model...")
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for features, labels in test_loader:
        outputs = model(features)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"✅ Test Accuracy: {accuracy:.2f}%")
# --------------------------------------------------
