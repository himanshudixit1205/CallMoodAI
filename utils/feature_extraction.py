import os
import librosa
import numpy as np
from tqdm import tqdm
import noisereduce as nr

# Paths
DATA_PATH = 'ravdess-data'
FEATURES_PATH = 'features'
os.makedirs(FEATURES_PATH, exist_ok=True)

# Parameters
SAMPLE_RATE = 22050
MFCC_DIM = 40
MAX_PAD_LEN = 128

# Emotion mapping
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

# Feature extraction
def extract_features(file_path, max_pad_len=MAX_PAD_LEN):
    try:
        audio, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)

        mfccs = librosa.feature.mfcc(
            y=audio,
            sr=sample_rate,
            n_mfcc=MFCC_DIM
        )

        # Pad / trim MFCC to fixed length
        if mfccs.shape[1] < max_pad_len:
            pad_width = max_pad_len - mfccs.shape[1]
            mfccs = np.pad(
                mfccs,
                pad_width=((0, 0), (0, pad_width)),
                mode='constant'
            )
        else:
            mfccs = mfccs[:, :max_pad_len]

        return mfccs[..., np.newaxis]
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


# Extract label from filename
def get_label_from_filename(filename):

    try:
        parts = filename.split('-')

        if len(parts) < 3:
            raise ValueError("Invalid filename format")

        emotion_code = parts[2]
        label = emotion_dict.get(emotion_code)

        if label is None:
            raise ValueError(f"Unknown emotion code: {emotion_code}")

        return label

    except Exception as e:
        print(f"[Error] Label extraction failed for {filename}: {e}")
        return None


# Build dataset
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

    print("Features and labels extracted")
    print("X shape:", X.shape)
    print("y shape:", y.shape)

    # Save processed dataset
    np.save(os.path.join(FEATURES_PATH, 'X.npy'), X)
    np.save(os.path.join(FEATURES_PATH, 'y.npy'), y)

    print(f"Saved to {FEATURES_PATH}/X.npy and y.npy")


if __name__ == '__main__':
    build_dataset()
