
import cv2
import mediapipe as mp
import os
import pandas as pd
import math

# --- Load labeled keyframes ---
keyframe_csv = "keyframes_labeled.csv"
df = pd.read_csv(keyframe_csv)

# --- Setup MediaPipe ---
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)
mp_drawing = mp.solutions.drawing_utils

# --- Feature Output ---
output_rows = []
output_columns = [
    "Video", "Frame", "Keyframe_Type",
    "Shoulder_X", "Shoulder_Y", "Elbow_X", "Elbow_Y", "Wrist_X", "Wrist_Y",
    "Hip_X", "Hip_Y", "Knee_X", "Knee_Y", "Ankle_X", "Ankle_Y",
    "Elbow_Angle", "Trunk_Lean_Angle", "Stride_Length",
    "Humerus_Straightness", "UpperArm_Compression", "Hand_Center_Offset"
]

def calculate_angle(a, b, c):
    ab = (a[0] - b[0], a[1] - b[1])
    cb = (c[0] - b[0], c[1] - b[1])
    dot = ab[0] * cb[0] + ab[1] * cb[1]
    mag_ab = math.hypot(*ab)
    mag_cb = math.hypot(*cb)
    if mag_ab == 0 or mag_cb == 0:
        return None
    return math.degrees(math.acos(dot / (mag_ab * mag_cb)))

# --- Process each labeled keyframe ---
for _, row in df.iterrows():
    video_path = os.path.join("videosJuly2", row["Video"])
    frame_index = int(row["Frame"])
    keyframe_type = row["Keyframe_Type"]

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    if not ret:
        continue

    h, w, _ = frame.shape
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)

    if not results.pose_landmarks:
        continue

    lm = results.pose_landmarks.landmark
    def px(p): return (lm[p].x * w, lm[p].y * h)

    shoulder = px(mp_pose.PoseLandmark.RIGHT_SHOULDER)
    elbow = px(mp_pose.PoseLandmark.RIGHT_ELBOW)
    wrist = px(mp_pose.PoseLandmark.RIGHT_WRIST)
    hip = px(mp_pose.PoseLandmark.RIGHT_HIP)
    knee = px(mp_pose.PoseLandmark.RIGHT_KNEE)
    ankle = px(mp_pose.PoseLandmark.RIGHT_ANKLE)
    left_hip = px(mp_pose.PoseLandmark.LEFT_HIP)
    left_ankle = px(mp_pose.PoseLandmark.LEFT_ANKLE)

    elbow_angle = calculate_angle(shoulder, elbow, wrist)
    trunk_angle = calculate_angle(shoulder, hip, left_hip)
    stride_length = math.hypot(left_ankle[0] - hip[0], left_ankle[1] - hip[1])
    humerus_straightness = calculate_angle(elbow, shoulder, hip)
    upper_arm_compression = math.hypot(shoulder[0] - elbow[0], shoulder[1] - elbow[1])
    hand_center_offset = abs((wrist[0] - (left_hip[0] + hip[0]) / 2))

    output_rows.append([
        row["Video"], frame_index, keyframe_type,
        shoulder[0], shoulder[1], elbow[0], elbow[1], wrist[0], wrist[1],
        hip[0], hip[1], knee[0], knee[1], ankle[0], ankle[1],
        round(elbow_angle, 2) if elbow_angle else None,
        round(trunk_angle, 2) if trunk_angle else None,
        round(stride_length, 2),
        round(humerus_straightness, 2) if humerus_straightness else None,
        round(upper_arm_compression, 2),
        round(hand_center_offset, 2)
    ])

# --- Save extracted features ---
features_df = pd.DataFrame(output_rows, columns=output_columns)
features_df.to_csv("keyframe_features_dataset.csv", index=False)
print("✅ Features extracted and saved to keyframe_features_dataset.csv")
