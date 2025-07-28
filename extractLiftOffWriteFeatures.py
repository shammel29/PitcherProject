import os
import pandas as pd
import numpy as np

# Paths
label_sheet = "LiftOffStandingUp - Sheet1.csv"
csv_folder = "media_pipe_outputs"
output_csv = "lift_off_features_labeled.csv"

# Load label sheet
df_labels = pd.read_csv(label_sheet)

# List to store extracted feature rows
data = []

for idx, row in df_labels.iterrows():
    video_name = row["Video_Name"].strip()
    lift_frame = row["LiftOff_Frame"]
    label = row["STANDING_UP"]

    base_name = os.path.splitext(video_name)[0]
    pose_csv_path = os.path.join(csv_folder, f"{base_name}.csv")

    if not os.path.exists(pose_csv_path):
        print(f"⚠️ Pose CSV not found: {pose_csv_path}")
        continue

    df_pose = pd.read_csv(pose_csv_path)
    df_pose.set_index("Frame", inplace=True)

    if lift_frame not in df_pose.index:
        print(f"⚠️ Frame {lift_frame} missing in {pose_csv_path}")
        continue

    features = df_pose.loc[lift_frame]
    features = features.to_dict()
    features["Label"] = label
    features["Video"] = video_name
    features["Frame"] = lift_frame

    data.append(features)

# Save as training dataset
df_out = pd.DataFrame(data)
df_out.to_csv(output_csv, index=False)
print(f"✅ Saved labeled feature CSV: {output_csv}")
