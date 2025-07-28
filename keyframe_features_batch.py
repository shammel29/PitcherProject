# Edited batch feature extraction script with additional features and normalization

import cv2
import mediapipe as mp
import os
import pandas as pd
import math
from tqdm import tqdm

# --- Load labeled keyframes ---
keyframe_csv = "keyframes_labeled.csv"
df_labels = pd.read_csv(keyframe_csv)

# --- Expand windows ---
window_map = {'MaxKneeLift': 2}
expanded = []
for _, row in df_labels.iterrows():
    w = window_map.get(row['Keyframe_Type'], 1)
    for off in range(-w, w+1):
        fn = row['Frame'] + off
        if fn >= 0:
            expanded.append({
                'Video': row['Video'],
                'Frame': fn,
                'Keyframe_Type': row['Keyframe_Type']
            })
df_labels_expanded = pd.DataFrame(expanded)

# --- Group by video ---
grouped = df_labels_expanded.groupby("Video")

# --- Setup MediaPipe ---
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)

# --- Prepare output ---
output = []

def angle(a, b, c):
    ab = (a[0] - b[0], a[1] - b[1])
    cb = (c[0] - b[0], c[1] - b[1])
    dp = ab[0]*cb[0] + ab[1]*cb[1]
    mag = math.hypot(*ab) * math.hypot(*cb)
    return math.degrees(math.acos(max(min(dp/mag,1),-1))) if mag else None

# --- Process each video once ---
for video, group in tqdm(grouped, desc="Videos"):
    video_path = os.path.join("smallVideoSet", video)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    for _, row in group.iterrows():
        frame_idx = int(row["Frame"])
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        h, w, _ = frame.shape
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(img)
        if not res.pose_landmarks:
            continue

        lm = res.pose_landmarks.landmark
        # helper to pixel coords
        def px(p): return (lm[p].x * w, lm[p].y * h)

        # Key landmarks
        r_sh = px(mp_pose.PoseLandmark.RIGHT_SHOULDER)
        l_sh = px(mp_pose.PoseLandmark.LEFT_SHOULDER)
        r_el = px(mp_pose.PoseLandmark.RIGHT_ELBOW)
        r_wr = px(mp_pose.PoseLandmark.RIGHT_WRIST)
        r_hp = px(mp_pose.PoseLandmark.RIGHT_HIP)
        r_knee = px(mp_pose.PoseLandmark.RIGHT_KNEE)
        l_knee = px(mp_pose.PoseLandmark.LEFT_KNEE)
        r_ank = px(mp_pose.PoseLandmark.RIGHT_ANKLE)
        l_ank = px(mp_pose.PoseLandmark.LEFT_ANKLE)
        r_toe = px(mp_pose.PoseLandmark.RIGHT_FOOT_INDEX)
        l_toe = px(mp_pose.PoseLandmark.LEFT_FOOT_INDEX)
        l_heel = px(mp_pose.PoseLandmark.LEFT_HEEL)

        # Torso reference distances
        torso_height = abs(r_sh[1] - r_hp[1]) or 1
        shoulder_width = abs(r_sh[0] - l_sh[0]) or 1

        # 1) Humerus angle (elbow-shoulder-hip)
        humerus_ang = angle(r_el, r_sh, r_hp)

        # 2) Femur angle (hip-knee-ankle)
        femur_ang = angle(r_hp, r_knee, r_ank)

        # 3) Right toe & left toe distance normalized (to torso height)
        r_toe_y_norm = r_toe[1] / torso_height
        l_toe_y_norm = l_toe[1] / torso_height

        # 4) Left heel normalized
        l_heel_y_norm = l_heel[1] / torso_height

        # 5) Left ankle normalized
        l_ankle_x_norm = l_ank[0] / torso_height
        l_ankle_y_norm = l_ank[1] / torso_height

        # 6) Right wrist vs. right shoulder horizontal alignment (abs y difference)
        horiz_align_rs_wr = abs(r_wr[1] - r_sh[1]) / torso_height

        # 7) Sternum point: midpoint between shoulders
        sternum_x = (r_sh[0] + l_sh[0]) / 2
        sternum_y = (r_sh[1] + l_sh[1]) / 2

        # 8) Sternum and wrist vertical alignment (abs x difference)
        vert_align_st_wr = abs(r_wr[0] - sternum_x) / shoulder_width

        # Existing features...
        # elbow angle, trunk lean, stride, etc.
        r_trunk_ang = angle(r_sh, r_hp, l_sh)
        stride = math.hypot(l_ank[0]-r_hp[0], l_ank[1]-r_hp[1])
        norm_stride = stride / torso_height
        comp = math.hypot(r_sh[0]-r_el[0], r_sh[1]-r_el[1])

        # Append all features
        output.append({
            "Video": video,
            "Frame": frame_idx,
            "Keyframe_Type": row["Keyframe_Type"],
            "Humerus_Angle": humerus_ang,
            "Femur_Angle": femur_ang,
            "R_Toe_Y_Norm": r_toe_y_norm,
            "L_Toe_Y_Norm": l_toe_y_norm,
            "L_Heel_Y_Norm": l_heel_y_norm,
            "L_Ankle_X_Norm": l_ankle_x_norm,
            "L_Ankle_Y_Norm": l_ankle_y_norm,
            "HorizAlign_ShoulderWrist": horiz_align_rs_wr,
            "VertAlign_SternumWrist": vert_align_st_wr,
            "Trunk_Lean_Angle": r_trunk_ang,
            "Norm_Stride": norm_stride,
            "Stride_Length": stride,
            "UpperArm_Compression": comp
        })

    cap.release()

# --- Save to CSV ---
df_out = pd.DataFrame(output)
df_out.to_csv("keyframe_features_enhanced.csv", index=False)
