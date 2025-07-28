import cv2
import numpy as np
import tensorflow as tf
import joblib
import mediapipe as mp
import math
import argparse

# ——— 1) Load your two models ———
# CNN classifier for LiftOff
img_model = tf.keras.models.load_model("liftoff_image_classifier.h5")
# RF classifier for final confirmation (expects the same features you trained on)
rf = joblib.load("keyframe_classifier.joblib")

# ——— 2) Prepare MediaPipe Pose ———
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ——— 3) Reuse your existing extract_features() ———
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


# ——— 4) Frame‐by‐frame loop ———
def main(video_path, img_th, rf_th, skip):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    prev_lm = None

    print(f"Processing {video_path} — {total} frames at {fps:.1f}fps")
    for i in range(0, total, skip):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break

        # ——— A) CNN filter ———
        img = cv2.resize(frame, (224,224))
        probs = img_model.predict(np.expand_dims(img,0), verbose=0)[0][0]
        if probs < img_th:
            continue

        # ——— B) Pose & Feature Extraction ———
        feats, prev_lm = extract_features(frame, prev_lm, fps)
        if feats is None:
            prev_lm = None
            continue

        # ——— C) RF confirm ———
        X = np.array([feats])
        proba = rf.predict_proba(X)[0]
        label = rf.classes_[proba.argmax()]
        conf  = proba.max()

        # only accept if RF agrees & is confident
        if label == "LiftOff" and conf >= rf_th:
            t = i / fps
            print(f"✔️ LiftOff @ frame {i} (~{t:.2f}s) [CNN={probs:.2f}, RF={conf:.2f}]")

    cap.release()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("video", help="Path to video file")
    p.add_argument("--img_th", type=float, default=0.6,
                   help="CNN probability threshold")
    p.add_argument("--rf_th", type=float, default=0.8,
                   help="RF probability threshold")
    p.add_argument("--skip", type=int, default=1,
                   help="Frame skip factor")
    args = p.parse_args()
    main(args.video, args.img_th, args.rf_th, args.skip)
