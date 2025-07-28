# batch_preprocess.py
import os
import json
import cv2
import math
import joblib
import argparse
import mediapipe as mp
import numpy as np
from tqdm import tqdm

# --- Load trained RF keyframe classifier ---
kf_model = joblib.load('keyframe_classifier.joblib')

# --- MediaPipe setup ---
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=True)

# --- Feature extraction (must match training) ---
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

# --- Cluster runs of the same label within ±2 frames, pick best confidence ---
def cluster_keyframes(raw, gap=2):
    clustered, curr = [], None
    for det in raw:
        f,l,c = det['frame'], det['label'], det['conf']
        if not curr or l!=curr['label'] or f>curr['end']+gap:
            curr = {'label':l, 'conf':c, 'start':f, 'end':f, 'best':f}
            clustered.append(curr)
        else:
            curr['end'] = f
            if c>curr['conf']:
                curr['conf'], curr['best'] = c, f
    return [
        {'label':c['label'], 'frame':c['best'], 'conf':c['conf']}
        for c in clustered
    ]

# --- Detect & cluster keyframes ---
def detect_keyframes(video_path, threshold=0.4, skip=1):
    cap   = cv2.VideoCapture(video_path)
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    raw, prev_lm = [], None
    for i in range(0, total, skip):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            continue

        feats, prev_lm = extract_features(frame, prev_lm, fps)
        if not feats:
            continue

        proba = kf_model.predict_proba([feats])[0]
        conf  = float(proba.max())
        if conf >= threshold:
            lab = kf_model.classes_[proba.argmax()]
            raw.append({'frame':i, 'label':lab, 'conf':conf})

    cap.release()
    events = cluster_keyframes(raw, gap=2)
    return raw, events

# --- Draw skeleton on top of a frame ---
def render_overlay(frame):
    img = frame.copy()
    res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if res.pose_landmarks:
        mp_draw.draw_landmarks(img, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    return img

# --- CLI entrypoint ---
if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--video',     '-v', required=True)
    p.add_argument('--outdir',    '-o', default='preproc')
    p.add_argument('--threshold','-t', type=float, default=0.4)
    p.add_argument('--skip',     '-s', type=int,   default=1)
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(os.path.join(args.outdir,'overlays'), exist_ok=True)

    raw, events = detect_keyframes(args.video,
                                   threshold=args.threshold,
                                   skip=args.skip)

    # dump JSON
    with open(os.path.join(args.outdir,'raw.json'),    'w') as f: json.dump(raw,    f, indent=2)
    with open(os.path.join(args.outdir,'events.json'),'w') as f: json.dump(events, f, indent=2)

    # save overlay images
    cap = cv2.VideoCapture(args.video)
    for ev in tqdm(events, desc='Rendering overlays'):
        fidx = ev['frame']
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ret, frm = cap.read()
        if not ret: continue
        ov = render_overlay(frm)
        fn = f"{fidx:06d}_{ev['label']}_{ev['conf']:.2f}.jpg"
        cv2.imwrite(os.path.join(args.outdir,'overlays',fn), ov)
    cap.release()

    print(f"Done: raw.json, events.json, and overlays/ in {args.outdir}")
