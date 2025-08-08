import streamlit as st
import librosa
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import noisereduce as nr
import io
from utils.feature_extraction import get_label_from_filename, extract_features

# Load trained model
model = tf.keras.models.load_model("saved_model/best_emotion_model.keras")

# Define class labels
class_labels = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

# Page title
st.title("🎙️ CallMoodAI")
st.write("Upload a `.wav` audio file and this app will detect the emotion in the speaker's voice.")

# File uploader
uploaded_file = st.file_uploader("Choose a .wav file", type=["wav"])

# Plot spectrogram
def plot_spectrogram(y, sr):
    fig, ax = plt.subplots()
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', ax=ax)
    ax.set_title('Spectrogram (dB)')
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    st.pyplot(fig)

# If file is uploaded
if uploaded_file is not None:
    # Play uploaded audio in the app
    st.audio(uploaded_file, format='audio/wav')
    
    with st.spinner("🔍 Processing audio and predicting emotion..."):
        # Extract features from uploaded audio
        features = extract_features(uploaded_file)  # shape will be (128, 128, 1)
        
        # Add batch dimension so model input becomes (1, 128, 128, 1)
        features = np.expand_dims(features, axis=0)
        
        # Predict emotion
        prediction = model.predict(features)
        predicted_index = np.argmax(prediction)
        emotion = class_labels[predicted_index]

    # Show original emotion from file name
    st.success(f"🎯 Original Emotion: **{get_label_from_filename(uploaded_file.name)}**")
    
    # Show predicted emotion
    st.success(f"🎯 Predicted Emotion: **{emotion.upper()}**")
        
    # Show top 3 predictions
    probs = prediction.flatten()
    top3_idx = probs.argsort()[-3:][::-1]
    st.subheader("📈 Top 3 Predictions:")
    for idx in top3_idx:
        st.write(f"{class_labels[idx].capitalize()}: {probs[idx]*100:.2f}%")

