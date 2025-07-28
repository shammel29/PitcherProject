import os
import pandas as pd
import numpy as np
import cv2

# --- SETTINGS ---
video_folder = "videos"
csv_folder = "media_pipe_outputs"
output_csv = "lift_off_features_labeled.csv"
os.makedirs("features_output", exist_ok=True)

# --- Smoothing Helper ---
def smooth(col, window=5):
    return np.convolve(col.interpolate(limit_direction="both"), np.ones(window)/window, mode='same')

# --- Output List ---
data_rows = []

# --- Process Each Video ---
for video_file in os.listdir(video_folder):
    if not video_file.endswith(".mp4"):
        continue

    base_name = os.path.splitext(video_file)[0]
    csv_path = os.path.join(csv_folder, f"{base_name}.csv")

    if not os.path.exists(csv_path):
        print(f"⚠️ Skipping {video_file}: no matching CSV")
        continue

    df = pd.read_csv(csv_path)
    if len(df) < 10:
        print(f"⚠️ Skipping {video_file}: not enough frames")
        continue

    # Smooth relevant columns
    for col in ["Hip_X", "Ankle_Y", "Wrist_Y", "Wrist_X", "Elbow_X", "Shoulder_X",
                "Left_Shoulder_X", "Right_Shoulder_X", "Right_Knee_X", "Right_Foot_Index_X"]:
        if col not in df.columns:
            print(f"❌ Column missing: {col} in {base_name}.csv")
            continue
        df[col] = smooth(df[col])

    # --- Detect Lift-Off ---
    hip_xs = df["Hip_X"]
    ankle_ys = df["Ankle_Y"]
    lift_off = None
    for i in range(8, len(df)):
        hip_dx = hip_xs[i] - hip_xs[i - 8]
        ankle_dy = ankle_ys[i - 8] - ankle_ys[i]
        if hip_dx > 4 and ankle_dy > 4:
            lift_off = int(df['Frame'].iloc[i])
            break

    if lift_off is None:
        print(f"❌ No Lift-Off detected in {video_file}")
        continue

    # --- Extract features at Lift-Off ---
    row = df[df["Frame"] == lift_off]
    if row.empty:
        print(f"❌ Frame {lift_off} not found in {base_name}.csv")
        continue

    row = row.iloc[0]

    # --- Compute Features ---
    left_shoulder_x = row["Left_Shoulder_X"]
    right_shoulder_x = row["Right_Shoulder_X"]
    sternum_x = (left_shoulder_x + right_shoulder_x) / 2

    knee_x = row["Right_Knee_X"]
    toe_x = row["Right_Foot_Index_X"]

    sternum_to_knee = sternum_x - knee_x
    toe_to_knee = toe_x - knee_x

    # --- Append to dataset ---
    data_rows.append({
        "Video": base_name,
        "LiftOff_Frame": lift_off,
        "Sternum_X": sternum_x,
        "Knee_X": knee_x,
        "Foot_X": toe_x,
        "Sternum_to_Knee_Dist": sternum_to_knee,
        "Toe_to_Knee_Dist": toe_to_knee,
        "Standing_Up": None  # ← You’ll fill this manually or using your label spreadsheet
    })

# --- Save Final CSV ---
output_df = pd.DataFrame(data_rows)
output_df.to_csv(output_csv, index=False)
print(f"✅ Done! Saved feature data to {output_csv}")
