import streamlit as st
import numpy as np
import pandas as pd
import librosa, soundfile as sf
import tensorflow as tf

# ------------------ 1. Load model & label map ------------------
MODEL_PATH   = "saved_models/audio_classification_CNN.keras"
LABELMAP_CSV = "label_map.csv"

model = tf.keras.models.load_model(MODEL_PATH)
label_map = pd.read_csv(LABELMAP_CSV)
idx_to_label = dict(zip(label_map['target'], label_map['category']))

# training-time parameters (must match!)
SR        = 22050
N_MELS    = 128
N_FFT     = 1024
HOP       = 512
DURATION  = 5.0
FIX_LEN   = int(DURATION * SR)

# mean & std you computed during training
mean = np.load("mel_mean.npy")
std  = np.load("mel_std.npy")

# ------------------ 2. Preprocessing ------------------
def wav_to_logmel(path):
    y, sr = librosa.load(path, sr=SR, mono=True)
    if len(y) < FIX_LEN:
        y = np.pad(y, (0, FIX_LEN - len(y)))
    else:
        y = y[:FIX_LEN]
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS,
                                       n_fft=N_FFT, hop_length=HOP)
    S_db = librosa.power_to_db(S, ref=np.max)
    return S_db.astype(np.float32)

def predict_clip(file):
    mel = wav_to_logmel(file)
    mel = ((mel - mean) / std)[..., None]
    mel = np.expand_dims(mel, 0)
    probs = model.predict(mel)[0]
    pred_idx = int(np.argmax(probs))
    return pred_idx, float(probs[pred_idx])

# ------------------ 3. Streamlit UI ------------------
st.title("🔊 Sound Classifier Using CNN")
st.write("Upload a `.wav` clip (≤5 s) to see the predicted category")

uploaded = st.file_uploader("Choose an audio file", type=["wav", "mp3"])

if uploaded:
    # save to temp so librosa can read
    with open("temp.wav", "wb") as f:
        f.write(uploaded.read())
    
    # play audio
    audio_data, samplerate = sf.read("temp.wav")
    st.audio("temp.wav")

    # run prediction
    pred_idx, conf = predict_clip("temp.wav")
    pred_label = idx_to_label[pred_idx]

    # Display only prediction text (no graph)
    st.subheader(f"Prediction: **{pred_label}** (confidence {conf:.2f}) ")
















