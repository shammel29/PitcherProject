import cv2
import argparse

def show_frame(video_path: str, frame_number: int):
    """
    Opens the video at `video_path`, seeks to `frame_number`, and displays that frame.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_number < 0 or frame_number >= total_frames:
        print(f"Error: frame_number {frame_number} out of range (0 to {total_frames-1})")
        cap.release()
        return

    # Seek to the desired frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"Error: Could not read frame {frame_number}")
        return

    cv2.imshow(f"Frame {frame_number}", frame)
    print(f"Showing frame {frame_number}. Press any key to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Show a specific frame from a video")
    parser.add_argument("video", help="Path to the video file")
    parser.add_argument("frame", type=int, help="Frame number to display (0-indexed)")
    args = parser.parse_args()

    show_frame(args.video, args.frame)
