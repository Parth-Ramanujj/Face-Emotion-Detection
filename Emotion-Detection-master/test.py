import av
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
from utils import process_frame

st.set_page_config(page_title="Live Emotion Detection", page_icon="🙂", layout="centered")

st.title("🎭 Live Emotion Detection")
st.markdown("Ensure your webcam is connected and allow browser access. Press **Start** to begin!")

def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")
    
    # Process the frame using our existing utils logic
    processed_img = process_frame(img)
    
    return av.VideoFrame.from_ndarray(processed_img, format="bgr24")

st.markdown("<small style='color: grey;'>*Note for iPhone Users: Do not open this link from WhatsApp/Instagram. Open it directly in the **Safari** app, otherwise the camera will not load.*</small>", unsafe_allow_html=True)

webrtc_streamer(
    key="emotion-detection",
    mode=WebRtcMode.SENDRECV,
    video_frame_callback=video_frame_callback,
    media_stream_constraints={
        "video": {"facingMode": "user"}, 
        "audio": False
    },
    rtc_configuration={
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
            {"urls": ["stun:stun2.l.google.com:19302"]},
            {"urls": ["stun:stun.stunprotocol.org:3478"]},
            {"urls": ["stun:stun.twilio.com:3478"]}
        ]
    },
    async_processing=True
)

