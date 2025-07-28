import cv2
import mediapipe as mp
import math
import matplotlib.pyplot as plt

# Setup MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Load video
cap = cv2.VideoCapture("resources/lucyPitch.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)  # Frame rate of the video

# Lists to store time and angular velocity
times = []
angular_velocities = []

prev_angle = None
frame_num = 0

def calculate_orientation_angle(left_shoulder, right_shoulder):
    """Returns angle of the shoulder line in degrees (relative to horizontal)."""
    dx = right_shoulder[0] - left_shoulder[0]
    dy = right_shoulder[1] - left_shoulder[1]
    angle_rad = math.atan2(dy, dx)
    return math.degrees(angle_rad)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        h, w, _ = frame.shape

        # Get left and right shoulders
        left = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_xy = (left.x * w, left.y * h)
        right_xy = (right.x * w, right.y * h)

        # Calculate chest rotation angle (from shoulder line)
        current_angle = calculate_orientation_angle(left_xy, right_xy)

        # Time in seconds for this frame
        time_sec = frame_num / fps
        times.append(time_sec)

        # Calculate angular velocity (deg/sec)
        if prev_angle is not None:
            delta_angle = current_angle - prev_angle

            # Ensure shortest rotation direction (account for wrap-around at 360°)
            if delta_angle > 180:
                delta_angle -= 360
            elif delta_angle < -180:
                delta_angle += 360

            angular_velocity = delta_angle * fps  # degrees per second
            angular_velocities.append(angular_velocity)
        else:
            angular_velocities.append(0)

        prev_angle = current_angle

    frame_num += 1

cap.release()

# ----- PLOT -----
plt.figure(figsize=(10, 6))
plt.plot(times, angular_velocities, label="Chest Angular Velocity (°/s)", color='blue')
plt.xlabel("Time (s)")
plt.ylabel("Angular Velocity (degrees/sec)")
plt.title("Chest Angular Velocity Over Time")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
