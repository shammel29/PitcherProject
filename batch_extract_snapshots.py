import os
import cv2
import math
import joblib
import argparse
import numpy as np
import pandas as pd
import mediapipe as mp
from tqdm import tqdm

# --- Load your model & set up MediaPipe once ---
model = joblib.load('keyframe_classifier_final.joblib')
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)

def extract_features(frame, prev_lm, fps):
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = pose.process(rgb)
    if not res.pose_landmarks:
        return None, prev_lm
    lm = res.pose_landmarks.landmark
    def px(p): return (lm[p].x*w, lm[p].y*h)
    # keypoints
    r_sh = px(mp_pose.PoseLandmark.RIGHT_SHOULDER)
    l_sh = px(mp_pose.PoseLandmark.LEFT_SHOULDER)
    r_el = px(mp_pose.PoseLandmark.RIGHT_ELBOW)
    r_wr = px(mp_pose.PoseLandmark.RIGHT_WRIST)
    r_hp = px(mp_pose.PoseLandmark.RIGHT_HIP)
    r_kn = px(mp_pose.PoseLandmark.RIGHT_KNEE)
    l_ank = px(mp_pose.PoseLandmark.LEFT_ANKLE)
    r_toe= px(mp_pose.PoseLandmark.RIGHT_FOOT_INDEX)
    l_toe= px(mp_pose.PoseLandmark.LEFT_FOOT_INDEX)
    l_heel=px(mp_pose.PoseLandmark.LEFT_HEEL)

    def angle(a,b,c):
        ab,cb = (a[0]-b[0],a[1]-b[1]),(c[0]-b[0],c[1]-b[1])
        dp = ab[0]*cb[0]+ab[1]*cb[1]
        mag = math.hypot(*ab)*math.hypot(*cb)
        return math.degrees(math.acos(max(min(dp/mag,1),-1))) if mag else 0

    torso_h = abs(r_sh[1]-r_hp[1]) or 1
    shoulder_w = abs(r_sh[0]-l_sh[0]) or 1

    feats = [
      # 1–2 angles
      angle(r_el,r_sh,r_hp),
      angle(r_hp,r_kn,l_ank),
      # 3–4 toe y’s
      r_toe[1]/torso_h, l_toe[1]/torso_h,
      # 5 heel y
      l_heel[1]/torso_h,
      # 6–7 left ankle
      l_ank[0]/torso_h, l_ank[1]/torso_h,
      # 8 horiz align
      abs(r_wr[1]-r_sh[1])/torso_h,
      # 9 vert align
      abs(r_wr[0]-((r_sh[0]+l_sh[0])/2))/shoulder_w,
      # 10 trunk lean
      angle(r_sh,r_hp,l_sh),
      # 11–12 stride norm & length
      (lambda s=math.hypot(l_ank[0]-r_hp[0], l_ank[1]-r_hp[1]): s/torso_h)(),
      math.hypot(l_ank[0]-r_hp[0], l_ank[1]-r_hp[1]),
      # 13 arm compression
      math.hypot(r_sh[0]-r_el[0], r_sh[1]-r_el[1])
    ]

    # wrap prev_lm update too
    return feats, res.pose_landmarks.landmark

def cluster_keyframes(raw):
    """raw = list of (frame,label,conf) sorted by frame"""
    clustered = []
    cur = None
    for f,l,c in raw:
        if not cur or l!=cur[0] or f>cur[3]+2:
            clustered.append([l,c,f,f])
            cur = clustered[-1]
            cur[0],cur[1],cur[2],cur[3] = l,c,f,f
        else:
            if c>cur[1]:
                cur[1],cur[3] = c,f
    return [(row[0], row[3], row[1]) for row in clustered]

def process_video(video_path, threshold, skip, snapshots_dir, rows):
    name = os.path.splitext(os.path.basename(video_path))[0]
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Detect raw keyframes
    prev_lm = None
    raw = []
    for i in range(0, total, skip):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret,frame = cap.read()
        if not ret: break
        feats, prev_lm = extract_features(frame, prev_lm, fps)
        if feats:
            df = pd.DataFrame([feats], columns=model.feature_names_in_)
            proba = model.predict_proba(df)[0]
            lab_idx = proba.argmax()
            if proba[lab_idx]>=threshold:
                raw.append((i, model.classes_[lab_idx], proba[lab_idx]))

    cap.release()
    stretched = cluster_keyframes(raw)
    best_per_label = {}
    for lbl, fr, conf in stretched:
        if lbl not in best_per_label or conf > best_per_label[lbl][1]:
            best_per_label[lbl] = (fr, conf)

    # this is now your final list of (label, frame, confidence)
    clustered = [(lbl, f, c) for lbl, (f, c) in best_per_label.items()]

    # Snapshot and row prep
    cap = cv2.VideoCapture(video_path)
    vid_snap_dir = os.path.join(snapshots_dir, name)
    os.makedirs(vid_snap_dir, exist_ok=True)

    for idx,(lbl,fr,conf) in enumerate(clustered):
        if fr<0 or fr>=total: continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, fr)
        ret,frm = cap.read()
        if not ret: continue
        fname = f"{name}_{idx:03d}_{lbl}_f{fr}.png"
        path = os.path.join(vid_snap_dir, fname)
        cv2.imwrite(path, frm)
        rows.append({
          "Video": name,
          "Snapshot": os.path.join(name,fname),
          "Label": lbl,
          "Frame": fr,
          "Confidence": conf,
          "Mechanic_Rating": "",
          "Comments": ""
        })
    cap.release()

if __name__=="__main__":
    p = argparse.ArgumentParser()
    p.add_argument("folder", help="Folder of .mp4 videos")
    p.add_argument("--threshold",type=float,default=0.4)
    p.add_argument("--skip",type=int,default=1)
    p.add_argument("--snapshots",default="snapshots")
    p.add_argument("--output",default="mechanics_label_template.csv")
    args = p.parse_args()

    videos = [f for f in os.listdir(args.folder) if f.endswith(".mp4")]
    rows=[]
    for vid in tqdm(videos, desc="Videos"):
        process_video(os.path.join(args.folder,vid),
                      args.threshold,args.skip,
                      args.snapshots,rows)

    pd.DataFrame(rows).to_csv(args.output,index=False)
    print(f"\nExtracted {len(rows)} snapshots from {len(videos)} videos.")
import warnings
warnings.filterwarnings(
    "ignore",
    message="SymbolDatabase.GetPrototype() is deprecated"
)