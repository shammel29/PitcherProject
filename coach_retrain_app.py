# coach_retrain_app.py

import streamlit as st
import tempfile, os, csv 
import cv2
from batch_preprocess import detect_keyframes, render_overlay, cluster_keyframes

st.set_page_config(page_title="Pitch Keyframe Coach-In-The-Loop", layout="wide")
st.title("⚾ Pitch Keyframe Snapshot & Feedback")

# Sidebar controls
threshold = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.8, 0.01)
skip      = st.sidebar.number_input("Frame skip (speed vs. accuracy)", 1, 10, 2)
feedback_csv = "coach_feedback.csv"

# ensure feedback file exists
if not os.path.exists(feedback_csv):
    with open(feedback_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["video","label","frame","conf","is_correct","true_label","notes"])

# Video uploader
video_file = st.file_uploader("Upload pitching video", type=["mp4","mov","avi"])
if not video_file:
    st.info("Please upload a video to begin.")
    st.stop()

# save to disk
tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4", mode="wb")
tmp.write(video_file.read())
video_path = tmp.name
video_name = os.path.basename(video_path)

# run detector once and cache
@st.cache_data(show_spinner=False)
def run_detector(path, thr, sk):
    raw, events = detect_keyframes(path, threshold=thr, skip=sk)
    best = {
      c["label"]: {"frame": c["frame"], "conf": c["conf"]}
      for c in cluster_keyframes(raw, gap=2)
    }
    # pre-render overlays
    cap = cv2.VideoCapture(path)
    overlays = {}
    for info in best.values():
        f = info["frame"]
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ret, frm = cap.read()
        if not ret: continue
        ov = render_overlay(frm)
        _, buf = cv2.imencode(".jpg", ov)
        overlays[f] = buf.tobytes()
    cap.release()
    return best, overlays

with st.spinner("Detecting keyframes…"):
    best, overlays = run_detector(video_path, threshold, skip)
st.success(f"Detected {len(best)} keyframes")

# prepare for paging
items = list(best.items())  # [ (label, {"frame":…, "conf":…}), … ]
n = len(items)
if "idx" not in st.session_state:
    st.session_state.idx = 0
if "fb" not in st.session_state:
    st.session_state.fb = {}  # will store feedback per idx

# navigation buttons
col1, col2, col3 = st.columns([1,6,1])
with col1:
    if st.button("◀️ Previous") and st.session_state.idx > 0:
        st.session_state.idx -= 1
with col3:
    if st.button("Next ▶️") and st.session_state.idx < n-1:
        st.session_state.idx += 1

# current snapshot
label, info = items[st.session_state.idx]
frm, conf = info["frame"], info["conf"]
img_bytes = overlays.get(frm)

st.markdown(f"### {label} — Frame {frm} (conf {conf:.2f})")
if img_bytes:
    st.image(img_bytes, use_container_width =True)
else:
    st.warning("⚠️ Couldn't render that frame.")

# —— coach feedback form —— 
st.markdown("#### Coach Feedback")
# radio correct / incorrect
is_corr = st.radio(
    "Is this classification correct?",
    ("✓ Correct","✗ Incorrect"),
    key=f"corr_{st.session_state.idx}"
)
true_lbl = label
if is_corr == "✗ Incorrect":
    true_lbl = st.text_input(
        "Enter the correct label:",
        value=label,
        key=f"true_{st.session_state.idx}"
    )
notes = st.text_area(
    "Notes (optional):",
    key=f"notes_{st.session_state.idx}",
    height=80
)

# store into session_state.fb on change
st.session_state.fb[st.session_state.idx] = {
    "video":     video_name,
    "label":     label,
    "frame":     frm,
    "conf":      conf,
    "is_correct": is_corr=="✓ Correct",
    "true_label": true_lbl,
    "notes":      notes
}

# —— Save all feedback —— 
if st.button("💾 Save all feedback to CSV"):
    with open(feedback_csv, "a", newline="") as f:
        writer = csv.writer(f)
        for fb in st.session_state.fb.values():
            writer.writerow([
                fb["video"], fb["label"], fb["frame"], fb["conf"],
                fb["is_correct"], fb["notes"]
            ])
    st.success(f"Saved {len(st.session_state.fb)} feedback rows to `{feedback_csv}`")

