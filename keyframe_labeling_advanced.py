
import cv2
import os

# --- SETTINGS ---
video_folder = "smallVideoSet"
output_csv = "keyframes_labeled.csv"
keyframe_map = {
    '1': "LiftOff",
    '2': "MaxKneeLift",
    '3': "DragStart",
    '4': "CirclePeak",
    '5': "FootPlant",
    '6': "ThreeQuarterArmCircle",
    '7': "Connection",
    '8': "Release"
}

# --- OUTPUT SETUP ---
if not os.path.exists(output_csv):
    with open(output_csv, "w") as f:
        f.write("Video,Frame,Keyframe_Type\n")

# --- HELPER ---
def write_keyframe(video_name, frame_num, keyframe_type):
    with open(output_csv, "a") as f:
        f.write(f"{video_name},{frame_num},{keyframe_type}\n")

# --- MANUAL LABELING TOOL ---
for filename in sorted(os.listdir(video_folder)):
    if not filename.endswith(".mp4"):
        continue

    video_path = os.path.join(video_folder, filename)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"❌ Could not open {filename}")
        continue

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"🎥 Reviewing {filename} ({total_frames} frames)")
    print("➡️ Use A/D to scrub. Press 1–8 to label keyframes. ENTER = next video. ESC = skip.")

    frame_num = 0
    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()
        cv2.putText(display, f"{filename} | Frame: {frame_num}", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        y_start = 80
        for i, (key, label) in enumerate(keyframe_map.items()):
            # Draw black outline first (shadow effect)
            cv2.putText(display, f"{key} = {label}", (30, y_start + i * 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
            # Then draw colored text on top
            cv2.putText(display, f"{key} = {label}", (30, y_start + i * 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

        cv2.imshow("Keyframe Labeler", display)

        key = cv2.waitKey(0)

        if key == 27:  # ESC
            break
        elif key == 13:  # ENTER - next video
            break
        elif key == ord('f'):
            frame_num = min(frame_num + 1, total_frames - 1)
        elif key == ord('a'):
            frame_num = max(frame_num - 1, 0)
        elif key == ord('g'):
            frame_num = min(frame_num + 5, total_frames - 1)
        elif key == ord('s'):
            frame_num = max(frame_num - 5, 0)
        elif key == ord('d'):
            frame_num = max(frame_num - 20, 0)
        elif key == ord('h'):
            frame_num = min(frame_num + 20, total_frames - 1)
        elif chr(key) in keyframe_map:
            write_keyframe(filename, frame_num, keyframe_map[chr(key)])
            print(f"✅ {keyframe_map[chr(key)]} saved at frame {frame_num}")

    cap.release()
    cv2.destroyAllWindows()
