import streamlit as st
import cv2
import numpy as np
from deepface import DeepFace
from PIL import Image

st.set_page_config(page_title="Face Expression Analyzer", layout="wide")

st.title("😃 Facial Expression Analyzer - Web App")
st.write("AI Powered Emotion Detection using Deep Learning")

img_file = st.camera_input("Capture Face")

if img_file:
    image = Image.open(img_file)
    img = np.array(image)

    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    result = DeepFace.analyze(
        rgb_img,
        actions=['emotion'],
        enforce_detection=False
    )

    emotion = result[0]['dominant_emotion']

    st.image(rgb_img, caption=f"Detected Emotion: {emotion.upper()}", use_column_width=True)
