import streamlit as st
import librosa
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from utils.feature_extraction import get_label_from_filename, extract_features

# Load trained emotion model
model = tf.keras.models.load_model("saved_model/best_emotion_model.keras")

# Emotion classes used by the model
class_labels = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

# Streamlit UI
st.title("🎙️ CallMoodAI")
st.write("Upload a `.wav` file to detect the speaker's emotion.")

# Upload audio file
uploaded_file = st.file_uploader("Choose a .wav file", type=["wav"])

# Plot spectrogram of the audio
def plot_spectrogram(y, sr):
    fig, ax = plt.subplots()
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', ax=ax)
    ax.set_title("Spectrogram (dB)")
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    st.pyplot(fig)

if uploaded_file is not None:

    # Play uploaded audio
    st.audio(uploaded_file, format='audio/wav')

    with st.spinner("Processing audio..."):

        # Extract audio features
        features = extract_features(uploaded_file)

        # Add batch dimension for model input
        features = np.expand_dims(features, axis=0)

        # Predict emotion
        prediction = model.predict(features)
        predicted_index = np.argmax(prediction)
        emotion = class_labels[predicted_index]

    # Show original emotion from filename
    st.success(f"Original Emotion: **{get_label_from_filename(uploaded_file.name)}**")

    # Show predicted emotion
    st.success(f"Predicted Emotion: **{emotion.upper()}**")

    # Show top 3 predictions
    probs = prediction.flatten()
    top3_idx = probs.argsort()[-3:][::-1]

    st.subheader("Top 3 Predictions")

    for idx in top3_idx:
        st.write(f"{class_labels[idx].capitalize()}: {probs[idx]*100:.2f}%")
