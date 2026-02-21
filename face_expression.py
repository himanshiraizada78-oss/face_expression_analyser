import streamlit as st
import numpy as np
from deepface import DeepFace
from PIL import Image

st.set_page_config(page_title="Face Expression Analyzer", layout="wide")

st.title("😃 Facial Expression Analyzer - Live Web App")
st.write("AI Powered Emotion Detection using Deep Learning")

frame = st.camera_input("📸 Capture your face")

if frame:
    img = Image.open(frame)
    img = np.array(img)

    with st.spinner("Analyzing emotion..."):
        result = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)
        emotion = result[0]['dominant_emotion']

    st.success(f"😃 Detected Emotion: **{emotion.upper()}**")
