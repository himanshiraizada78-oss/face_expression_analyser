import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration
import cv2
import av
import numpy as np
from deepface import DeepFace
import logging
import os
os.environ["WEBRTC_IP_HANDLING_POLICY"] = "default"

# ---------------- LOGGING CONFIG ----------------
logging.basicConfig(
    filename="emotion_app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logging.info("Application Started")

# ---------------- UI CONFIG ----------------
st.set_page_config(page_title="Face Expression AI", layout="wide")
st.title("😃 Facial Expression Analyzer")
st.write("Real-time Emotion Detection using DeepFace & WebRTC")

# ---------------- WEBRTC CONFIG ----------------
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302",
                              "stun:stun1.l.google.com:19302"]}]}
)

# ---------------- VIDEO CALLBACK ----------------
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")

    try:
        results = DeepFace.analyze(
            img, actions=['emotion'], enforce_detection=False
        )

        emotion = results[0]['dominant_emotion']

        logging.info(f"Emotion Detected: {emotion}")

        cv2.putText(img, emotion.upper(), (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0), 2, cv2.LINE_AA)

    except Exception as e:
        logging.error(f"Detection Error: {str(e)}")

    return av.VideoFrame.from_ndarray(img, format="bgr24")

# ---------------- UI LAYOUT ----------------
col1, col2 = st.columns([2, 1])

with col1:
    webrtc_streamer(
        key="emotion-detection",
        video_frame_callback=video_frame_callback,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

with col2:
    st.subheader("Instructions")
    st.markdown("""
    1. Click **Start** to open your webcam.
    2. Ensure your face is well-lit.
    3. The AI will label your dominant emotion on the video.
    """)

    st.warning("Note: Initial loading may take a few seconds.")

logging.info("Webcam Stream Initialized")

