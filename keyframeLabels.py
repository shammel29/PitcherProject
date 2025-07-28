import cv2
import numpy as np
import mediapipe as mp
from scipy.signal import find_peaks

def detect_liftoff_rule(video_path, smooth_window=5, velocity_threshold=None):
    # 1) extract normalized ankle‐y over all frames
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True)
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    ys = []
    torso_refs = []
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        h, w, _ = frame.shape
        res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not res.pose_landmarks:
            ys.append(np.nan)
            torso_refs.append(np.nan)
            continue
        
        lm = res.pose_landmarks.landmark
        # back ankle y, and torso height = shoulder_y – hip_y
        r_ank_y = lm[mp_pose.PoseLandmark.RIGHT_ANKLE].y * h
        sh_y   = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * h
        hip_y  = lm[mp_pose.PoseLandmark.RIGHT_HIP].y * h
        torso_h = abs(sh_y - hip_y) or 1
        
        ys.append(r_ank_y / torso_h)
        torso_refs.append(torso_h)
    
    cap.release()
    ys = np.array(ys)
    
    # 2) fill gaps & smooth
    isnan = np.isnan(ys)
    ys[isnan] = np.interp(np.flatnonzero(isnan), np.flatnonzero(~isnan), ys[~isnan])
    # moving average
    kernel = np.ones(smooth_window) / smooth_window
    ys_smooth = np.convolve(ys, kernel, mode="same")
    
    # 3) compute upward velocity = negative derivative
    vel = -(ys_smooth[1:] - ys_smooth[:-1]) * fps  # shape = len(ys)-1
    
    # 4) dynamic threshold
    if velocity_threshold is None:
        mu, sigma = np.mean(vel), np.std(vel)
        velocity_threshold = mu + 2*sigma
    
    # 5) find peaks in vel above threshold
    peaks, props = find_peaks(vel, height=velocity_threshold, distance=int(fps*0.1))
    # distance=fps*0.1 to avoid multiple peaks within 0.1 s
    
    if len(peaks)==0:
        return None
    # first peak index + 1 (because vel is one shorter than ys)
    liftoff_frame = peaks[0] + 1
    return liftoff_frame, vel, velocity_threshold

# ——— Usage:
video = "path/to/pitch.mp4"
res = detect_liftoff_rule(video)
if res:
    frame_idx, velocity_series, thresh = res
    print(f"Detected LiftOff at frame {frame_idx} (threshold={thresh:.2f})")
else:
    print("No LiftOff found.")
