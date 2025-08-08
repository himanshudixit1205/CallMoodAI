# CallMoodAI 🎯  
Audio Emotion Detection System using Deep Learning  

## 📌 Overview  
**CallMoodAI** is an AI-powered application that detects emotions from speech recordings.  
It uses **MFCC-based feature extraction** and a **deep learning model** to classify emotions like Happy, Sad, Angry, Neutral, and more.  
The app runs on **Streamlit** for a simple and interactive user interface.

---

## ✨ Features  
- 🎙 **Emotion Detection from Audio** (`.wav` format)  
- 📊 **Top 3 Predictions** with confidence scores  
- ⚡ **Fast Processing** using pre-trained deep learning model  
- 🖥 **Streamlit Web App** for easy interaction  
- 🎯 Supports **real-time file uploads**  

---

## 🛠 Tech Stack  
- **Python 3.x**  
- **TensorFlow / Keras** – Deep Learning Model  
- **Librosa** – Audio Processing & Feature Extraction  
- **NumPy / Pandas** – Data Handling  
- **Streamlit** – Web Application Interface  

---

## 🚀 Installation  

1️⃣ **Clone the Repository**  
```bash
git clone https://github.com/himanshudixit1205/CallMoodAI.git
cd CallMoodAI
```

2️⃣ **Install Dependencies**  
```bash
pip install -r requirements.txt
```

---

## ▶️ Usage  

**Run the Streamlit App**  
```bash
streamlit run streamlit_app.py
```

**Steps:**  
1. Upload a `.wav` audio file.  
2. Wait for the app to process the file.  
3. View the **original** and **predicted** emotions along with confidence scores.  

---

## 📂 Project Structure  
```
CallMoodAI/
│-- streamlit_app.py                # Streamlit app entry point
│-- model/                # Trained model files
│-- utils/                # Helper functions
│-- requirements.txt      # Python dependencies
│-- README.md              # Project documentation
```

---

## 📊 Example Output  
```
Original Emotion: Happy  
Predicted Emotion: Happy (95.2%)  

Top 3 Predictions:
1. Happy: 95.2%  
2. Neutral: 3.4%  
3. Angry: 1.4%  
```

---

## 📜 License  
This project is licensed under the **MIT License** – you are free to use, modify, and distribute it.

---

## 🤝 Contributing  
Pull requests are welcome!  
If you find a bug or want a feature, open an **issue** or submit a **PR**.

---

## 📧 Contact  
**Author:** Himanshu Dixit  
**GitHub:** [himanshudixit1205](https://github.com/himanshudixit1205)  

