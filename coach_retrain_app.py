import streamlit as st
import tempfile
import cv2
from batch_preprocess import detect_keyframes, render_overlay, cluster_keyframes

st.set_page_config(page_title="Pitch Keyframe Reviewer", layout="wide")
st.title("⚾ Pitch Keyframe Reviewer (Optimized)")

threshold = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.8, 0.01)
skip = st.sidebar.number_input("Frame skip", 1, 10, 2)

video_file = st.file_uploader("Upload pitching video", type=["mp4", "mov", "avi"])
if not video_file:
    st.info("Upload a video to begin analysis.")
    st.stop()

# Save uploaded video to a temp file
tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4", mode="wb")
tmp.write(video_file.read())
video_path = tmp.name

# ------------------------------
# ✅ Cache heavy detection step
# ------------------------------
@st.cache_data(show_spinner=True)
def process_video_once(path, thr, sk):
    raw, _ = detect_keyframes(path, threshold=thr, skip=sk)
    clustered = cluster_keyframes(raw, gap=2)
    
    overlays = []
    cap = cv2.VideoCapture(path)
    for ev in clustered:
        cap.set(cv2.CAP_PROP_POS_FRAMES, ev["frame"])
        ret, frame = cap.read()
        if not ret:
            continue
        ov = render_overlay(frame)
        _, buf = cv2.imencode(".jpg", ov)
        overlays.append({
            "label": ev["label"],
            "frame": ev["frame"],
            "conf": ev["conf"],
            "img": buf.tobytes()
        })
    cap.release()
    return overlays

# Store results in session state
if "snapshots" not in st.session_state:
    with st.spinner("Processing video…"):
        st.session_state.snapshots = process_video_once(video_path, threshold, skip)
    st.session_state.index = 0

snapshots = st.session_state.snapshots

# ------------------------------
# ✅ Navigation buttons
# ------------------------------
col1, col2, col3 = st.columns([1, 3, 1])
with col1:
    if st.button("⬅️ Prev") and st.session_state.index > 0:
        st.session_state.index -= 1
with col3:
    if st.button("Next ➡️") and st.session_state.index < len(snapshots) - 1:
        st.session_state.index += 1

# ------------------------------
# ✅ Display current snapshot
# ------------------------------
current = snapshots[st.session_state.index]
st.image(current["img"], caption=f"{current['label']} | Frame {current['frame']} | Conf {current['conf']:.2f}")
