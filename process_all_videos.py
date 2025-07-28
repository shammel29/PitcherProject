import cv2
import mediapipe as mp
import math
import csv
import os

# Setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Paths
video_folder = "videosJuly2"
output_folder = "media_pipe_outputs"
os.makedirs(output_folder, exist_ok=True)

# Process all videos
for filename in os.listdir(video_folder):
    if not filename.endswith(".mp4"):
        continue

    video_path = os.path.join(video_folder, filename)
    cap = cv2.VideoCapture(video_path)
    output_csv_path = os.path.join(output_folder, filename.replace(".mp4", ".csv"))

    csv_file = open(output_csv_path, "w", newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "Frame",
        "Left_Shoulder_X", "Left_Shoulder_Y",
        "Right_Shoulder_X", "Right_Shoulder_Y",
        "Right_Knee_X", "Right_Knee_Y",
        "Right_Foot_Index_X", "Right_Foot_Index_Y",
        "Hip_X", "Hip_Y",
        "Ankle_X", "Ankle_Y",
        "Elbow_X", "Elbow_Y",
        "Wrist_X", "Wrist_Y"
    ])

    frame_num = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(img_rgb)

        if results.pose_landmarks:
            lms = results.pose_landmarks.landmark

            def get_xy(landmark):
                return lms[landmark].x * w, lms[landmark].y * h

            l_shoulder = get_xy(mp_pose.PoseLandmark.LEFT_SHOULDER)
            r_shoulder = get_xy(mp_pose.PoseLandmark.RIGHT_SHOULDER)
            r_knee = get_xy(mp_pose.PoseLandmark.RIGHT_KNEE)
            r_foot = get_xy(mp_pose.PoseLandmark.RIGHT_FOOT_INDEX)
            hip = get_xy(mp_pose.PoseLandmark.RIGHT_HIP)
            ankle = get_xy(mp_pose.PoseLandmark.LEFT_ANKLE)
            elbow = get_xy(mp_pose.PoseLandmark.RIGHT_ELBOW)
            wrist = get_xy(mp_pose.PoseLandmark.RIGHT_WRIST)

            csv_writer.writerow([
                frame_num,
                l_shoulder[0], l_shoulder[1],
                r_shoulder[0], r_shoulder[1],
                r_knee[0], r_knee[1],
                r_foot[0], r_foot[1],
                hip[0], hip[1],
                ankle[0], ankle[1],
                elbow[0], elbow[1],
                wrist[0], wrist[1]
            ])

        frame_num += 1

    cap.release()
    csv_file.close()
    print(f"✅ Processed {filename}")
