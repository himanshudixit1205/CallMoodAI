import os
import librosa
import numpy as np
from tqdm import tqdm
import noisereduce as nr

# Set your dataset directory here
DATA_PATH = 'ravdess-data'
FEATURES_PATH = 'features'
os.makedirs(FEATURES_PATH, exist_ok=True)

# Set parameters
SAMPLE_RATE = 22050
MFCC_DIM = 40
MAX_PAD_LEN = 128 

# Mapping RAVDESS emotion codes to emotion labels
emotion_dict = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

# Extract Features
def extract_features(file_path, max_pad_len=MAX_PAD_LEN):
    try:
        audio, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)
        
        # MFCC = Mel-Frequency Cepstral Coefficients — features that represent the short-term power spectrum of audio, mimicking how humans perceive sound.
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=MFCC_DIM)

        # It ensures all MFCC feature arrays have the same length (max_pad_len = 128) by:
        if mfccs.shape[1] < max_pad_len:
            pad_width = max_pad_len - mfccs.shape[1]
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfccs = mfccs[:, :max_pad_len] # Trimming
        return mfccs
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Filename
def get_label_from_filename(filename):
    try:
        parts = filename.split('-')
        if len(parts) < 3:
            raise ValueError("Filename format invalid")

        emotion_code = parts[2]
        label = emotion_dict.get(emotion_code)

        if label is None:
            raise ValueError(f"Unknown emotion code: {emotion_code}")

        return label

    except Exception as e:
        print(f"[Error] Failed to extract label from {filename}: {e}")
        return None


# Dataset
def build_dataset():
    print("\nExtracting features from audio files...")
    X, y = [], []

    for root, _, files in os.walk(DATA_PATH):
        for file in files:
            if file.endswith(".wav"):
                file_path = os.path.join(root, file)
                features = extract_features(file_path)

                if features is not None:
                    X.append(features)
                    y.append(get_label_from_filename(file))

    X = np.array(X)
    y = np.array(y)

    print("Features and labels extracted.")
    print("X shape:", X.shape)
    print("y shape:", y.shape)

    # Storing the processed features and labels in a fast, ready-to-load binary format so we don’t have to re-extract them every time.
    np.save(os.path.join(FEATURES_PATH, 'X.npy'), X)
    np.save(os.path.join(FEATURES_PATH, 'y.npy'), y)
    print(f"Saved to {FEATURES_PATH}/X.npy and y.npy")

if __name__ == '__main__':
    build_dataset()