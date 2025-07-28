
import cv2
import os

# --- SETTINGS ---
video_folder = "testVidForLab"
output_csv = "manual_keyframes_labeled.csv"
keyframe_label = "LiftOff"

# Define all evaluator tags
evaluator_tags = [

    "Standing_Up", "Staying_Open", "Marching", "Staying_In_Front",
    "Landing_Early", "Anchoring", "Locking", "Drifting",
    "Pushing", "Sweeping", "Leaning"
]

# --- OUTPUT SETUP ---
if not os.path.exists(output_csv):
    with open(output_csv, "w") as f:
        header = "Video,Keyframe,Frame," + ",".join(evaluator_tags) + "\n"
        f.write(header)

# --- HELPER ---
def write_keyframe(video_name, frame_num, labels):
    with open(output_csv, "a") as f:
        row = f"{video_name},{keyframe_label},{frame_num}," + ",".join(labels) + "\n"
        f.write(row)

# --- MANUAL REVIEW TOOL ---
for filename in sorted(os.listdir(video_folder)):
    if not filename.endswith(".mp4"):
        continue

    video_path = os.path.join(video_folder, filename)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"❌ Failed to open {filename}")
        continue

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"📹 Reviewing: {filename} ({total_frames} frames)")
    print("➡️ Use arrow keys to scrub. Press SPACE to label frame. ENTER to finish video. ESC to skip.")

    frame_num = 0
    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()
        cv2.putText(display, f"{filename} | Frame: {frame_num}", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.imshow("Label Frame", display)

        key = cv2.waitKey(0)

        if key == 27:  # ESC
            break
        elif key == 13:  # ENTER to finish video
            break
        elif key == 32:  # SPACE to label current frame
            labels = []
            current_tag_index = 0

            while current_tag_index < len(evaluator_tags):
                tag = evaluator_tags[current_tag_index]
                frame_copy = display.copy()
                cv2.putText(frame_copy, f"Label: {tag}", (30, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                cv2.putText(frame_copy, "Press 1 = Yes, 0 = No, Enter = Skip", (30, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
                cv2.imshow("Label Frame", frame_copy)
                tag_key = cv2.waitKey(0)

                if tag_key == ord('1'):
                    labels.append("1")
                    current_tag_index += 1
                elif tag_key == ord('0'):
                    labels.append("0")
                    current_tag_index += 1
                elif tag_key == 13:  # Enter to skip
                    labels.append("")
                    current_tag_index += 1
                elif tag_key == 27:  # ESC to cancel labeling
                    labels = []
                    break

            if labels:
                write_keyframe(filename, frame_num, labels)
                print(f"✅ Saved frame {frame_num} with labels: {labels}")
            else:
                print("⚠️ Labeling canceled.")

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

    cap.release()
    cv2.destroyAllWindows()
