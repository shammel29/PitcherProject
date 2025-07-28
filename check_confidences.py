# check_confidences.py
import cv2
import numpy as np
import joblib
import argparse
from detect_keyframes import extract_features  # your extractor

def main(video_path, threshold=None):
    # Load model
    model = joblib.load('keyframe_classifier.joblib')
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    
    # Track best (conf, frame) per class
    best = {lbl: (0.0, None) for lbl in model.classes_}
    
    prev_lm = None
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        feats, prev_lm = extract_features(frame, prev_lm, fps)
        if feats:
            print("DEBUG: feats length =", len(feats))
    # then wrap in DF...

        if feats is None:
            frame_idx += 1
            continue
        
        proba = model.predict_proba(np.array([feats]))[0]
        for lbl, p in zip(model.classes_, proba):
            # if this class’s prob is higher than any seen before, record it
            if p > best[lbl][0]:
                best[lbl] = (p, frame_idx)
        
        frame_idx += 1
    
    cap.release()
    
    # Print results
    print(f"Processed {frame_idx} frames from {video_path}\n")
    print("Peak confidence & frame per class:")
    for lbl, (p, f) in best.items():
        print(f"  {lbl:25s} → {p:.3f}  at frame {f}")
    
    if threshold is not None:
        print("\nClasses below threshold:")
        for lbl, (p, f) in best.items():
            if p < threshold:
                print(f"  {lbl:25s} → {p:.3f}  (never reached {threshold})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Find peak probabilities and their frames in a video"
    )
    parser.add_argument("video", help="Path to video file")
    parser.add_argument(
        "--threshold", type=float,
        help="Optional threshold to flag classes under this confidence"
    )
    args = parser.parse_args()
    main(args.video, args.threshold)

