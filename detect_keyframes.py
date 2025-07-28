# detect_keyframes.py
import cv2
import numpy as np
import joblib
import mediapipe as mp
import math
import argparse
import pandas as pd
from tqdm import tqdm

# 1) Load trained model
model = joblib.load('keyframe_classifier.joblib')

# 2) Set up MediaPipe pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)

# 3) Feature extraction matching training
def extract_features(frame, prev_lm, fps):
    h, w, _ = frame.shape
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = pose.process(img)
    if not res.pose_landmarks:
        return None, None
    lm = res.pose_landmarks.landmark
    
    def px(p): return (lm[p].x*w, lm[p].y*h)
    # keypoints
    r_sh = px(mp_pose.PoseLandmark.RIGHT_SHOULDER)
    l_sh = px(mp_pose.PoseLandmark.LEFT_SHOULDER)
    r_el = px(mp_pose.PoseLandmark.RIGHT_ELBOW)
    r_wr = px(mp_pose.PoseLandmark.RIGHT_WRIST)
    r_hp = px(mp_pose.PoseLandmark.RIGHT_HIP)
    r_kn = px(mp_pose.PoseLandmark.RIGHT_KNEE)
    l_kn = px(mp_pose.PoseLandmark.LEFT_KNEE)
    r_ank = px(mp_pose.PoseLandmark.RIGHT_ANKLE)
    l_ank = px(mp_pose.PoseLandmark.LEFT_ANKLE)
    r_toe = px(mp_pose.PoseLandmark.RIGHT_FOOT_INDEX)
    l_toe = px(mp_pose.PoseLandmark.LEFT_FOOT_INDEX)
    l_heel= px(mp_pose.PoseLandmark.LEFT_HEEL)
    
    def angle(a,b,c):
        ab = (a[0]-b[0],a[1]-b[1])
        cb = (c[0]-b[0],c[1]-b[1])
        dp = ab[0]*cb[0]+ab[1]*cb[1]
        mag = math.hypot(*ab)*math.hypot(*cb)
        return math.degrees(math.acos(max(min(dp/mag,1),-1))) if mag else 0
    
    # compute features
    torso_h = abs(r_sh[1]-r_hp[1]) or 1
    shoulder_w = abs(r_sh[0]-l_sh[0]) or 1
    
    feats = []
    # 1) Humerus_Angle
    feats.append(angle(r_el, r_sh, r_hp))
    # 2) Femur_Angle
    feats.append(angle(r_hp, r_kn, r_ank))
    # 3) R_Toe_Y_Norm, L_Toe_Y_Norm
    feats.append(r_toe[1]/torso_h)
    feats.append(l_toe[1]/torso_h)
    # 4) L_Heel_Y_Norm
    feats.append(l_heel[1]/torso_h)
    # 5) L_Ankle_X_Norm, L_Ankle_Y_Norm
    feats.append(l_ank[0]/torso_h)
    feats.append(l_ank[1]/torso_h)
    # 6) HorizAlign_ShoulderWrist
    feats.append(abs(r_wr[1]-r_sh[1])/torso_h)
    # 7) VertAlign_SternumWrist
    stern_x = (r_sh[0]+l_sh[0])/2
    feats.append(abs(r_wr[0]-stern_x)/shoulder_w)
    # 8) Trunk_Lean_Angle
    feats.append(angle(r_sh, r_hp, l_sh))
    # 9) Norm_Stride & Stride_Length
    stride = math.hypot(l_ank[0]-r_hp[0], l_ank[1]-r_hp[1])
    feats.append(stride/torso_h)
    # 10) UpperArm_Compression
    feats.append(math.hypot(r_sh[0]-r_el[0], r_sh[1]-r_el[1]))
    # 11) Elbow_Angular_Velocity
    if prev_lm:
        prev_sh = (prev_lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].x*w,
                   prev_lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].y*h)
        prev_el = (prev_lm[mp_pose.PoseLandmark.RIGHT_ELBOW].x*w,
                   prev_lm[mp_pose.PoseLandmark.RIGHT_ELBOW].y*h)
        prev_wr = (prev_lm[mp_pose.PoseLandmark.RIGHT_WRIST].x*w,
                   prev_lm[mp_pose.PoseLandmark.RIGHT_WRIST].y*h)
        prev_ang = angle(prev_el, prev_sh, prev_wr)
        feats.append((feats[0]-prev_ang)*fps)
    else:
        feats.append(0)
    
    return feats, res.pose_landmarks.landmark

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--skip", type=int, default=1,
                        help="Process every Nth frame to speed up")
    args = parser.parse_args()
    
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    prev_lm = None
    keyframes = []
    # progress bar
    for i in tqdm(range(0, total, args.skip), desc="Detecting frames"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break
        feats, prev_lm = extract_features(frame, prev_lm, fps)
        if feats:
            df = pd.DataFrame([feats], columns=model.feature_names_in_)
            proba = model.predict_proba(df)[0]
            label = model.classes_[proba.argmax()]
            conf = proba.max()
            if conf >= args.threshold:
                keyframes.append((i, label, conf))
    cap.release()
    
    # cluster
    clustered = []
    curr = None
    for f,l,c in keyframes:
        if not curr or l!=curr[0] or f>curr[1]+2:
            clustered.append([l,f,c,f,f])
            curr = (l,f,c,f,f)
        else:
            if c>clustered[-1][2]:
                clustered[-1][1] = f
                clustered[-1][2] = c
                clustered[-1][3] = f
    # save
    import csv
    with open('keyframes_clustered.csv','w',newline='') as f:
        w=csv.writer(f)
        w.writerow(['Label','BestFrame','Confidence'])
        for row in clustered:
            w.writerow([row[0], row[3], row[2]])
    print("Done. Results in keyframes_clustered.csv")

if __name__ == "__main__":
    main()
