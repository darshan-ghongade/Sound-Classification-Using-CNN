'''
import streamlit as st
import numpy as np
import pandas as pd
import librosa, soundfile as sf
import tensorflow as tf
from streamlit_webrtc import webrtc_streamer, WebRtcMode, ClientSettings
import av
import tempfile


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
    probs = model.predict(mel, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    return pred_idx, float(probs[pred_idx])

# ------------------ 3. Streamlit UI ------------------
st.title("🔊 Sound Classifier Using CNN")
st.write("Upload a `.wav` clip (≤5 s) OR record audio to see the predicted category")

# --- Option 1: Upload ---
uploaded = st.file_uploader("Choose an audio file", type=["wav", "mp3"])

# --- Option 2: Record ---
st.write("Or record a new audio:")
audio = audiorecorder("🎙️ Start Recording", "⏹️ Stop Recording")

# Decide which input to use
if uploaded or (len(audio) > 0):
    if uploaded:
        # Save uploaded file
        with open("temp.wav", "wb") as f:
            f.write(uploaded.read())
        file_path = "temp.wav"
    else:
        # Save recorded audio
        file_path = "recorded.wav"
        audio.export(file_path, format="wav")

    # Play audio
    st.audio(file_path)

    # Run prediction
    pred_idx, conf = predict_clip(file_path)
    pred_label = idx_to_label[pred_idx]

    st.subheader(f"Prediction: **{pred_label}**  (confidence {conf:.2f})")
'''

import streamlit as st
import numpy as np
import pandas as pd
import librosa, soundfile as sf
import tensorflow as tf
import tempfile
from streamlit_webrtc import webrtc_streamer, WebRtcMode, ClientSettings
import av

# ------------------ 1. Load model & label map ------------------
MODEL_PATH   = "saved_models/audio_classification_CNN.keras"
LABELMAP_CSV = "label_map.csv"

model = tf.keras.models.load_model(MODEL_PATH)
label_map = pd.read_csv(LABELMAP_CSV)
idx_to_label = dict(zip(label_map['target'], label_map['category']))

# Training-time parameters (must match!)
SR        = 22050
N_MELS    = 128
N_FFT     = 1024
HOP       = 512
DURATION  = 5.0
FIX_LEN   = int(DURATION * SR)

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
    probs = model.predict(mel, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    return pred_idx, float(probs[pred_idx])

# ------------------ 3. Streamlit UI ------------------
st.title("🔊 Sound Classifier Using CNN")
st.write("Upload a `.wav` clip (≤5 s) OR record audio to see the predicted category")

uploaded = st.file_uploader("Choose an audio file", type=["wav", "mp3"])

# Recording setup
st.write("Or record new audio (click 'Start' then 'Stop' to save):")

# Temporary file to store recording
temp_audio_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)

# Collect recorded frames
recorded_frames = []

def audio_frame_callback(frame: av.AudioFrame):
    # Append audio frame to global list
    recorded_frames.append(frame)

webrtc_ctx = webrtc_streamer(
    key="sound-recorder",
    mode=WebRtcMode.SENDONLY,
    client_settings=ClientSettings(
        media_stream_constraints={"audio": True, "video": False}
    ),
    audio_frame_callback=audio_frame_callback,
    async_processing=True,
)

# If user recorded something and pressed "Process"
if webrtc_ctx.state.playing and st.button("Use Recorded Audio"):
    # Combine frames into a single file
    if recorded_frames:
        # Convert frames to mono float32 PCM
        audio_arrays = [frame.to_ndarray().mean(axis=1) for frame in recorded_frames]
        audio_data = np.concatenate(audio_arrays)
        sf.write(temp_audio_file.name, audio_data, recorded_frames[0].sample_rate)
        file_path = temp_audio_file.name
    else:
        file_path = None
else:
    file_path = None

if uploaded:
    with open("temp.wav", "wb") as f:
        f.write(uploaded.read())
    file_path = "temp.wav"

# Run prediction if we have a file
if file_path:
    st.audio(file_path)
    pred_idx, conf = predict_clip(file_path)
    pred_label = idx_to_label[pred_idx]
    st.subheader(f"Prediction: **{pred_label}**  (confidence {conf:.2f})")




