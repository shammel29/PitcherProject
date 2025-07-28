import cv2
import mediapipe as mp
import math
import csv

# ----- SETUP -----
# These tools let us track body positions in video frames
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Load a video of a pitcher (can be swapped for any pitching video)
cap = cv2.VideoCapture("resources/lucyPitch.mp4")

# Create a CSV (spreadsheet) where we’ll save movement data for each frame
csv_file = open("pitch_data.csv", "w", newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow([
    "Frame",
    "Shoulder_X", "Shoulder_Y",
    "Elbow_X", "Elbow_Y",
    "Wrist_X", "Wrist_Y",
    "Elbow_Angle",
    "Stride_Length",
    "Trunk_Lean_Angle",
    "Wrist_Hip_Distance"
])

# ----- ANGLE FUNCTION -----
def calculate_angle(a, b, c):
    """
    Calculates the angle at joint 'b' given 3 points (like shoulder, elbow, wrist).
    For example, this is how we find elbow angle at release.
    """
    ab = (a[0] - b[0], a[1] - b[1])  # vector from elbow to shoulder
    cb = (c[0] - b[0], c[1] - b[1])  # vector from elbow to wrist
    dot_product = ab[0]*cb[0] + ab[1]*cb[1]
    mag_ab = math.hypot(*ab)
    mag_cb = math.hypot(*cb)
    if mag_ab == 0 or mag_cb == 0:
        return None
    angle_rad = math.acos(dot_product / (mag_ab * mag_cb))
    return math.degrees(angle_rad)  # convert to degrees

# ----- MAIN LOOP -----
frame_num = 0  # Counts how many frames we’ve analyzed

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # End of video

    # Convert image to format MediaPipe uses
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)  # Run pose tracking on this frame

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark  # Get all joint positions

        # We'll focus on the RIGHT arm for now (change if analyzing a lefty)
        shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
        wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]

        # Convert the joint positions from percentages to pixel locations
        h, w, _ = frame.shape
        shoulder_xy = (shoulder.x * w, shoulder.y * h)
        elbow_xy = (elbow.x * w, elbow.y * h)
        wrist_xy = (wrist.x * w, wrist.y * h)

        # Elbow angle (important for understanding arm slot, extension, timing)
        angle = calculate_angle(shoulder_xy, elbow_xy, wrist_xy)

        # Also track stride length, trunk lean, and extension
        hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
        ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]  # front foot
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]

        # Convert these to pixel coordinates too
        hip_xy = (hip.x * w, hip.y * h)
        ankle_xy = (ankle.x * w, ankle.y * h)
        left_shoulder_xy = (left_shoulder.x * w, left_shoulder.y * h)

        # Stride length = distance between front ankle and back hip
        stride_length = math.hypot(ankle_xy[0] - hip_xy[0], ankle_xy[1] - hip_xy[1])

        # Trunk lean = angle between the shoulders and the hip (forward tilt)
        trunk_lean_angle = calculate_angle(left_shoulder_xy, shoulder_xy, hip_xy)

        # Wrist-to-hip distance = how far the throwing hand is from the body
        wrist_hip_distance = math.hypot(wrist_xy[0] - hip_xy[0], wrist_xy[1] - hip_xy[1])

        # Save all these metrics to our CSV so we can review or graph later
        csv_writer.writerow([
            frame_num,
            shoulder_xy[0], shoulder_xy[1],
            elbow_xy[0], elbow_xy[1],
            wrist_xy[0], wrist_xy[1],
            round(angle, 2) if angle else "NA",
            round(stride_length, 2),
            round(trunk_lean_angle, 2),
            round(wrist_hip_distance, 2)
        ])

        # Draw body lines and joints on the video frame so we can visualize it
        mp_drawing.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Display the elbow angle directly on the video for real-time feedback
        if angle:
            cv2.putText(frame, f"Elbow Angle: {int(angle)}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)

    # Show the video frame (with joints and angle text) in a window
    cv2.imshow("Pitch Tracker", frame)

    # Press 'q' to quit watching early
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

    frame_num += 1  # Move to the next frame

# ----- CLEANUP -----
cap.release()        # Stop reading video
csv_file.close()     # Save and close the spreadsheet file
cv2.destroyAllWindows()  # Close the display window
