import streamlit as st
import tempfile, os
import cv2
from batch_preprocess import detect_keyframes, render_overlay, cluster_keyframes

st.set_page_config(page_title="Pitch Keyframe Reviewer", layout="wide")
st.title("⚾ Pitch Keyframe Reviewer (Optimized)")

# Sidebar controls
threshold = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.8, 0.01)
skip = st.sidebar.number_input("Frame skip", 1, 10, 2)

# Upload video
video_file = st.file_uploader("Upload pitching video", type=["mp4", "mov", "avi"])
if not video_file:
    st.info("Please upload a video to start.")
    st.stop()

# Save uploaded video
tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4", mode="wb")
tmp.write(video_file.read())
video_path = tmp.name

# Preprocess video ONCE and cache results
@st.cache_resource(show_spinner=True)
def preprocess_video(path, thr, sk):
    # Detect keyframes
    raw, _ = detect_keyframes(path, threshold=thr, skip=sk)
    clustered = cluster_keyframes(raw, gap=2)

    cap = cv2.VideoCapture(path)
    overlays = []
    for c in clustered:
        frame_idx = c["frame"]
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frm = cap.read()
        if not ret:
            continue
        overlay_img = render_overlay(frm)
        overlays.append({
            "label": c["label"],
            "frame": frame_idx,
            "conf": c["conf"],
            "img": cv2.imencode(".jpg", overlay_img)[1].tobytes()
        })
    cap.release()
    return overlays

with st.spinner("Processing video... This will only run once."):
    processed_frames = preprocess_video(video_path, threshold, skip)

# Navigation
if "idx" not in st.session_state:
    st.session_state.idx = 0

total_frames = len(processed_frames)

col1, col2, col3 = st.columns([1, 3, 1])
with col1:
    if st.button("⬅️ Previous") and st.session_state.idx > 0:
        st.session_state.idx -= 1
with col3:
    if st.button("Next ➡️") and st.session_state.idx < total_frames - 1:
        st.session_state.idx += 1

# Display current frame
if total_frames > 0:
    current = processed_frames[st.session_state.idx]
    st.image(current["img"], caption=f"{current['label']} (Frame {current['frame']}, Conf {current['conf']:.2f})", use_column_width=True)
else:
    st.warning("No keyframes detected.")
