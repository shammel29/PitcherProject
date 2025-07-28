#!/usr/bin/env python3
# extract_confirmed_features.py

import os
import cv2
import math
import argparse
import pandas as pd
import mediapipe as mp

mp_pose = mp.solutions.pose
pose   = mp_pose.Pose(static_image_mode=True)

def angle(a, b, c):
    # compute angle at b between points a–b–c
    ab = (a[0]-b[0], a[1]-b[1])
    cb = (c[0]-b[0], c[1]-b[1])
    dp = ab[0]*cb[0] + ab[1]*cb[1]
    mag = math.hypot(*ab) * math.hypot(*cb) or 1e-6
    return math.degrees(math.acos(max(min(dp/mag, 1), -1)))

def extract_for_row(video_dir, row):
    video_path = os.path.join(video_dir, row.Video)
    # row.Video is just the basename (no “.mp4”), so try common extensions:
    base = row.Video
    video_path = None
    for ext in (".mp4", ".mov", ".avi"):
        cand = os.path.join(video_dir, base + ext)
        if os.path.isfile(cand):
            video_path = cand
            break
    if video_path is None:
        print(f"⚠️  Video file not found for {base} (tried .mp4/.mov/.avi)")
        return None
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(row.Frame))
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print(f"⚠️  Could not read frame {row.Frame} of {row.Video}")
        return None

    h, w, _ = frame.shape
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = pose.process(img)
    if not res.pose_landmarks:
        print(f"⚠️  No pose landmarks at {row.Video}@{row.Frame}")
        return None

    lm = res.pose_landmarks.landmark
    def px(p): return (lm[p].x*w, lm[p].y*h)

    # keypoints
    r_sh = px(mp_pose.PoseLandmark.RIGHT_SHOULDER)
    l_sh = px(mp_pose.PoseLandmark.LEFT_SHOULDER)
    r_el = px(mp_pose.PoseLandmark.RIGHT_ELBOW)
    r_wr = px(mp_pose.PoseLandmark.RIGHT_WRIST)
    r_hp = px(mp_pose.PoseLandmark.RIGHT_HIP)
    r_kn = px(mp_pose.PoseLandmark.RIGHT_KNEE)
    l_ank= px(mp_pose.PoseLandmark.LEFT_ANKLE)
    r_toe= px(mp_pose.PoseLandmark.RIGHT_FOOT_INDEX)
    l_toe= px(mp_pose.PoseLandmark.LEFT_FOOT_INDEX)
    l_heel=px(mp_pose.PoseLandmark.LEFT_HEEL)

    torso_h    = abs(r_sh[1] - r_hp[1]) or 1
    shoulder_w = abs(r_sh[0] - l_sh[0]) or 1

    # 13 features
    feats = {
      "Elbow_Angle":        angle(r_el, r_sh, r_hp),
      "Femur_Angle":        angle(r_hp, r_kn, l_ank),
      "R_Toe_Y_Norm":       r_toe[1]/torso_h,
      "L_Toe_Y_Norm":       l_toe[1]/torso_h,
      "L_Heel_Y_Norm":      l_heel[1]/torso_h,
      "L_Ankle_X_Norm":     l_ank[0]/torso_h,
      "L_Ankle_Y_Norm":     l_ank[1]/torso_h,
      "HorizAlign_ShoulderWrist":
                            abs(r_wr[1] - r_sh[1]) / torso_h,
      "VertAlign_SternumWrist":
                            abs(r_wr[0] - ((r_sh[0]+l_sh[0])/2)) / shoulder_w,
      "Trunk_Lean_Angle":   angle(r_sh, r_hp, l_sh),
      "Norm_Stride":        math.hypot(l_ank[0]-r_hp[0], l_ank[1]-r_hp[1]) / torso_h,
      "Stride_Length":      math.hypot(l_ank[0]-r_hp[0], l_ank[1]-r_hp[1]),
      "UpperArm_Compression":
                            math.hypot(r_sh[0]-r_el[0], r_sh[1]-r_el[1])
    }

    return {
      "Video": row.Video,
      "Frame": row.Frame,
      "Keyframe_Type": row.Label,
      **feats
    }

def main():
    p = argparse.ArgumentParser(
        description="Extract features at confirmed keyframes"
    )
    p.add_argument("--video_dir", required=True,
                   help="Folder containing your .mp4 videos")
    p.add_argument("--csv",       required=True,
                   help="mechanics_confirmed.csv from the review UI")
    p.add_argument("--out_csv",   default="features_confirmed.csv",
                   help="Where to write the extracted features")
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    rows = []
    for _, row in df.iterrows():
        rec = extract_for_row(args.video_dir, row)
        if rec:
            rows.append(rec)

    pd.DataFrame(rows).to_csv(args.out_csv, index=False)
    print(f"✅ Wrote {len(rows)} feature rows to {args.out_csv}")

if __name__ == "__main__":
    main()
