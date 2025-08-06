import tkinter as tk
from tkinter import messagebox
import sounddevice as sd
import soundfile as sf
import torch
import numpy as np
import librosa
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Define constants
SAMPLE_RATE = 22050
DURATION = 3
FILENAME = 'live_input.wav'

# Placeholder: Load your trained model, scaler, and label encoder here
# model = torch.load('your_model.pth')
# scaler = joblib.load('scaler.pkl')
# le = joblib.load('label_encoder.pkl')

# Dummy placeholders for demo (remove when using real models)
class DummyModel(torch.nn.Module):
    def forward(self, x):
        return torch.randn((1, 4))  # Fake output (4 emotion classes)

model = DummyModel()
scaler = StandardScaler()
le = LabelEncoder()
le.classes_ = np.array(['angry', 'happy', 'neutral', 'sad'])  # example classes

# Extract features (using MFCCs)
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_mean = np.mean(mfcc.T, axis=0)
    return mfcc_mean

# Function to record and predict
def record_and_predict():
    try:
        status_label.config(text="🎤 Recording...")
        root.update()
        
        recording = sd.rec(int(SAMPLE_RATE * DURATION), samplerate=SAMPLE_RATE, channels=1)
        sd.wait()
        sf.write(FILENAME, recording, SAMPLE_RATE)

        features = extract_features(FILENAME)
        features = scaler.fit_transform([features])  # NOTE: replace with scaler.transform([features]) in real case
        tensor = torch.tensor(features, dtype=torch.float32)

        model.eval()
        with torch.no_grad():
            output = model(tensor)
            predicted = torch.argmax(output, dim=1)
            emotion = le.inverse_transform(predicted.numpy())[0]
            status_label.config(text=f"🎯 Predicted Emotion: {emotion}")

    except Exception as e:
        messagebox.showerror("Error", str(e))
        status_label.config(text="❌ Error occurred.")

# GUI setup
root = tk.Tk()
root.title("Voice Emotion Detector")
root.geometry("300x200")
root.resizable(False, False)

title_label = tk.Label(root, text="🎙️ Emotion Predictor", font=("Helvetica", 16))
title_label.pack(pady=10)

record_button = tk.Button(root, text="Record & Predicting...", command=record_and_predict, bg="green", fg="white", font=("Helvetica", 12))
record_button.pack(pady=20)

status_label = tk.Label(root, text="Press the button to start", font=("Helvetica", 10))
status_label.pack()

root.mainloop()
