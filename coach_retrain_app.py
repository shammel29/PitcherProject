import streamlit as st
import tempfile
import cv2
import os
import csv
from batch_preprocess import detect_keyframes, render_overlay, cluster_keyframes

st.set_page_config(page_title="Pitch Keyframe Reviewer", layout="wide")
st.title("⚾ Pitch Keyframe Reviewer for Retraining")

threshold = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.8, 0.01)
skip = st.sidebar.number_input("Frame skip", 1, 10, 2)

feedback_file = "coach_feedback.csv"
# Ensure CSV exists
if not os.path.exists(feedback_file):
    with open(feedback_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["video", "frame", "predicted_label", "is_correct", "true_label", "notes"])

video_file = st.file_uploader("Upload pitching video", type=["mp4", "mov", "avi"])
if not video_file:
    st.info("Upload a video to begin analysis.")
    st.stop()

# Save video to temp
tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4", mode="wb")
tmp.write(video_file.read())
video_path = tmp.name
video_name = os.path.basename(video_file.name)

# Cache detection
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

# Session state
if "snapshots" not in st.session_state:
    with st.spinner("Processing video…"):
        st.session_state.snapshots = process_video_once(video_path, threshold, skip)
    st.session_state.index = 0

snapshots = st.session_state.snapshots
current = snapshots[st.session_state.index]

# Navigation
col1, col2, col3 = st.columns([1, 3, 1])
with col1:
    if st.button("⬅️ Prev") and st.session_state.index > 0:
        st.session_state.index -= 1
with col3:
    if st.button("Next ➡️") and st.session_state.index < len(snapshots) - 1:
        st.session_state.index += 1

# Display snapshot
st.image(current["img"], caption=f"{current['label']} | Frame {current['frame']} | Conf {current['conf']:.2f}")

# --------------------------
# 🟢 Coach Feedback Section
# --------------------------
st.subheader("Coach Feedback")
correct = st.radio("Was the model prediction correct?", ["✅ Correct", "❌ Incorrect"], horizontal=True)
true_label = current["label"]
if correct == "❌ Incorrect":
    true_label = st.text_input("Enter the correct label:", value=current["label"])

notes = st.text_area("Additional notes:", height=80)

if st.button("💾 Save Feedback"):
    with open(feedback_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            video_name,
            current["frame"],
            current["label"],
            correct == "✅ Correct",
            true_label,
            notes
        ])
    st.success("✅ Feedback saved! Move to next frame.")

# Display progress
st.sidebar.markdown(f"**Progress:** {st.session_state.index+1} / {len(snapshots)} frames reviewed")
