import os
import pandas as pd
import numpy as np
import cv2

# --- SETTINGS ---
video_folder = "videos"
csv_folder = "media_pipe_outputs"
output_folder = "output_frames"
os.makedirs(output_folder, exist_ok=True)

# --- Smoothing helper ---
def smooth(col, window=5):
    return np.convolve(col.interpolate(limit_direction="both"), np.ones(window)/window, mode='same')

# --- Process each video ---
for video_file in os.listdir(video_folder):
    if not video_file.endswith(".mp4"):
        continue

    base_name = os.path.splitext(video_file)[0]
    video_path = os.path.join(video_folder, video_file)
    csv_path = os.path.join(csv_folder, f"{base_name}.csv")

    if not os.path.exists(csv_path):
        print(f"⚠️ Skipping {video_file}: no matching CSV")
        continue

    print(f"🎬 Processing {video_file}")

    df = pd.read_csv(csv_path)
    if len(df) < 10:
        print(f"⚠️ Skipping {video_file}: not enough frames")
        continue

    # Smooth tracking
    hip_xs = smooth(df["Hip_X"])
    ankle_ys = smooth(df["Ankle_Y"])
    wrist_ys = smooth(df["Wrist_Y"])
    wrist_xs = smooth(df["Wrist_X"])
    elbow_xs = smooth(df["Elbow_X"])
    shoulder_xs = smooth(df["Shoulder_X"])

    # Detect LIFTOFF
    lift_off = None
    for i in range(8, len(df)):
        hip_dx = hip_xs[i] - hip_xs[i - 8]
        ankle_dy = ankle_ys[i - 8] - ankle_ys[i]
        if hip_dx > 4 and ankle_dy > 4:
            lift_off = int(df['Frame'].iloc[i])
            print(f"✅ LIFTOFF at frame {lift_off}")
            break

    # Detect CIRCLE PEAK
    circle_peak = int(df['Wrist_Y'].idxmin())
    circle_peak = int(df['Frame'].iloc[circle_peak])

    # Detect RELEASE
    release = None
    if circle_peak is not None:
        for i in range(circle_peak + 1, len(df)):
            wx, wy = df['Wrist_X'].iloc[i], df['Wrist_Y'].iloc[i]
            ex = df['Elbow_X'].iloc[i]
            sx = df['Shoulder_X'].iloc[i]

            wrist_ahead = wx > ex and wx > sx

            if i >= 3:
                wy_prev = df['Wrist_Y'].iloc[i - 3]
                wrist_drop = wy_prev - wy > 5
            else:
                wrist_drop = False

            if wrist_ahead and wrist_drop:
                release = int(df['Frame'].iloc[i])
                print(f"✅ RELEASE at frame {release}")
                break

    # Map keyframes
    keyframes = {
        "liftoff": lift_off,
        "circle_peak": circle_peak,
        "release": release
    }

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Failed to open {video_file}")
        continue

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        for name, fnum in keyframes.items():
            if fnum is not None and frame_idx == fnum:
                label = name.upper()
                cv2.putText(frame, f"{label} (frame {fnum})", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                save_path = os.path.join(output_folder, f"{base_name}_{name}.jpg")
                cv2.imwrite(save_path, frame)
                print(f"✅ Saved: {save_path}")

        frame_idx += 1

    cap.release()

print("✅ All videos processed.")
