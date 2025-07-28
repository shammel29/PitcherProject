import streamlit as st
import tempfile, cv2
from batch_preprocess import detect_keyframes, render_overlay
import numpy as np

st.set_page_config(layout="wide")
st.title("Pitch Evaluation")

# Sidebar
threshold = st.sidebar.slider("Thresh", 0.0, 1.0, 0.8, 0.01)
skip      = st.sidebar.number_input("Skip frames", 1, 10, 2)

# 1) Upload
video_file = st.file_uploader("Upload video", type=["mp4","avi","mov"])
if not video_file:
    st.stop()

# 2) Save to temp
tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
tmp.write(video_file.read())
video_path = tmp.name

# 3) Preprocess everything—detect + grab overlays + grab raw frames
@st.cache_data(show_spinner=False)
def preprocess_all(path, thr, sk):
    raw, events = detect_keyframes(path, threshold=thr, skip=sk)
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Dicts for O(1) lookup
    overlay_bytes = {}
    raw_bytes     = {}
    
    # First, pre-render overlays for event frames
    for ev in events:
        f = ev["frame"]
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ret, frm = cap.read()
        if not ret: continue
        ov = render_overlay(frm)
        _, buf = cv2.imencode(".jpg", ov)
        overlay_bytes[f] = buf.tobytes()
    
    # Then, pre-render *all* raw frames as JPEG (this sucks for very long vids,
    # but if your goals are snappiness it may be OK up to a couple hundred frames)
    for i in range(total):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frm = cap.read()
        if not ret: break
        _, buf = cv2.imencode(".jpg", frm)
        raw_bytes[i] = buf.tobytes()
    
    cap.release()
    return events, overlay_bytes, raw_bytes

with st.spinner("Analyzing…"):
    events, overlays, raws = preprocess_all(video_path, threshold, skip)
st.success(f"Done – {len(events)} events found")

# 4) Slider + display
total_frames = len(raws)
col1, col2 = st.columns([3,1])
with col1:
    frame_idx = st.slider("Frame", 0, total_frames-1, 0)
    # super-cheap lookup, no OpenCV, no MediaPipe
    img_to_show = overlays.get(frame_idx, raws[frame_idx])
    st.image(img_to_show, use_column_width=True)

with col2:
    st.header("Events")
    if not events:
        st.write("No keyframes detected")
    else:
        sel = st.selectbox(
            "Go to event",
            range(len(events)),
            format_func=lambda i: f"{events[i]['label']} @ {events[i]['frame']}"
        )
        ev = events[sel]
        st.markdown(f"**Label:** {ev['label']}  ")
        st.markdown(f"**Frame:** {ev['frame']}  ")
        st.markdown(f"**Conf:** {ev['conf']:.2f}")
