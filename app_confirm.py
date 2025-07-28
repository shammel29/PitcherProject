# coach_retrain_app.py

import streamlit as st
import tempfile, json, os, csv
import pandas as pd
from batch_preprocess import detect_keyframes, render_overlay

st.set_page_config(layout="wide")
st.title("⚾ Pitch Analyzer & Coach Feedback")

# — Sidebar controls —
threshold = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.8, 0.01)
skip      = st.sidebar.number_input("Frame skip", 1, 10, 2)
feedback_file = "coach_feedback.csv"

# Ensure feedback CSV exists
if not os.path.exists(feedback_file):
    with open(feedback_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["video","frame","predicted_label","is_correct","true_label","notes"])

# — Video uploader —
uploaded = st.file_uploader("1) Upload a new pitching video", type=["mp4","mov","avi"])
if not uploaded:
    st.info("Upload a video to begin analysis.")
    st.stop()

# Save upload to disk
tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4", mode="wb")
tmp.write(uploaded.read())
video_path = tmp.name

# — Preprocess: detect & overlay, cached so it only runs once per upload/params —
@st.cache_data(show_spinner=False)
def run_detector(path, thr, sk):
    raw, events = detect_keyframes(path, threshold=thr, skip=sk)
    # load video once
    import cv2
    cap = cv2.VideoCapture(path)
    overlays = {}
    for ev in events:
        f = ev["frame"]
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ret, frm = cap.read()
        if not ret: continue
        ov = render_overlay(frm)
        _, buf = cv2.imencode(".jpg", ov)
        overlays[f] = buf.tobytes()
    cap.release()
    return events, overlays

with st.spinner("Detecting keyframes…"):
    events, overlays = run_detector(video_path, threshold, skip)
st.success(f"Detected {len(events)} events")

# — Show list of events & feedback form —
st.sidebar.header("2) Coach Feedback")
feedback_rows = []
for i, ev in enumerate(events):
    st.sidebar.markdown(f"**Event {i+1}:** {ev['label']} @ frame {ev['frame']} (conf {ev['conf']:.2f})")
    is_corr = st.sidebar.radio(f"Correct?", ("✓ Correct","✗ Incorrect"), key=f"cb_{i}")
    true_lbl = ev['label']
    if is_corr == "✗ Incorrect":
        true_lbl = st.sidebar.text_input(f"True label:", value=ev['label'], key=f"tl_{i}")
    notes = st.sidebar.text_area(f"Notes:", key=f"nt_{i}", height=50)
    feedback_rows.append({
        "video": os.path.basename(video_path),
        "frame": ev["frame"],
        "predicted_label": ev["label"],
        "is_correct": is_corr=="✓ Correct",
        "true_label": true_lbl,
        "notes": notes
    })

if st.sidebar.button("Save Feedback"):
    with open(feedback_file, "a", newline="") as f:
        writer = csv.writer(f)
        for row in feedback_rows:
            writer.writerow([
                row["video"], row["frame"], row["predicted_label"],
                row["is_correct"], row["true_label"], row["notes"]
            ])
    st.sidebar.success("Feedback saved!")

# — Main viewer: scrub through frames —
st.header("3) Video Preview")
total = len(overlays) and max(overlays.keys())+1 or 0
frame_idx = st.slider("Frame", 0, total-1, 0)
img = overlays.get(frame_idx)
if img:
    st.image(img, use_column_width=True)
else:
    st.warning("No overlay for this frame")
