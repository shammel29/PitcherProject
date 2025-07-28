import os
import pandas as pd
import numpy as np
import cv2

# --- SETTINGS ---
video_folder = "videos"
csv_folder = "media_pipe_outputs"
output_folder = "liftoff_frames"
os.makedirs(output_folder, exist_ok=True)

# --- Smoothing helper ---
def smooth(col, window=5):
    return np.convolve(col.interpolate(limit_direction="both"), np.ones(window)/window, mode='same')

# --- Output CSV Setup ---
output_csv_path = os.path.join(output_folder, "liftoff_keyframes.csv")
with open(output_csv_path, "w") as f:
    f.write("Video,Frame\n")

# --- Process Each Video ---
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

    # Detect LIFTOFF
    lift_off = None
    for i in range(8, len(df)):
        hip_dx = hip_xs[i] - hip_xs[i - 8]
        ankle_dy = ankle_ys[i - 8] - ankle_ys[i]
        if hip_dx > 4 and ankle_dy > 7:
            lift_off = int(df['Frame'].iloc[i])+5
            print(f"✅ LIFTOFF at frame {lift_off}")
            break

    if lift_off is None:
        print(f"❌ No LIFTOFF detected for {video_file}")
        continue

    # Write to CSV
    with open(output_csv_path, "a") as f:
        f.write(f"{video_file},{lift_off}\n")

    # Save frames
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for offset in range(-5, 6):  # 5 frames before to 5 after
        target_frame = lift_off + offset
        if target_frame < 0 or target_frame >= total_frames:
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ret, frame = cap.read()
        if not ret:
            continue

        tag = "liftoff" if offset == 0 else f"liftoff_offset{offset}"
        filename = f"{base_name}_{tag}_frame{target_frame}.jpg"
        save_path = os.path.join(output_folder, filename)
        cv2.imwrite(save_path, frame)

    cap.release()

print("✅ Done extracting LiftOff frames and saving CSV.")
