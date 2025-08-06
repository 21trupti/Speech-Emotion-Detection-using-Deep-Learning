# Speech-Emotion-Detection-using-Deep-Learning
The main objective of the Speech Emotion Recognition project is to develop a system that can accurately detect and classify human emotions based on speech signals. The system will analyze various features from audio recordings like pitch, tone, intensity . It helps to categorize emotions like happiness, sadness, anger, surprise, fear, and neutral.

📁 Speech Emotion Recognition 🎙️
This project implements a deep learning-based Speech Emotion Recognition (SER) system using PyTorch. It extracts MFCC features from .wav audio files using Librosa, and trains a neural network to classify emotions such as happy, sad, angry, etc.

🔧 Features:
Audio feature extraction (MFCC)

Label encoding and data normalization

Custom PyTorch Dataset & DataLoader

Fully connected neural network for emotion classification

Model training and test evaluation with accuracy reporting

🧠 Tech Stack:
Python, PyTorch, Librosa, scikit-learn

📌 Features
🔉 Audio Feature Extraction: Extracts MFCCs from .wav audio files using Librosa.
🔄 Data Preprocessing:
Label encoding with LabelEncoder
Feature normalization using StandardScaler
Train-test split with stratified sampling
🧠 Model Architecture:
Fully connected feedforward neural network using PyTorch
Includes ReLU activation and dropout for regularization
📊 Training & Evaluation:
Uses CrossEntropyLoss and Adam optimizer
Evaluates model performance on unseen test data
Calculates overall accuracy

🚀 How to Run
Install dependencies:

bash
pip install torch librosa scikit-learn numpy matplotlib seaborn
Set your audio directory in the script:

python
AUDIO_DIR = r"C:\path\to\your\audio\folder"

bash
python train.py





