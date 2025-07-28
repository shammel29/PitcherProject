#!/usr/bin/env python3
import cv2
import numpy as np
import mediapipe as mp
import math
import argparse
from scipy.signal import find_peaks

def moving_average(x, w):
    return np.convolve(x, np.ones(w)/w, mode="same")

def detect_liftoff(frames, fps, smooth_w=5, sigma_mul=1.5, min_dist_frames=5):
    # ankle_y_norm → back-ankle vertical normalized by torso
    y = np.array([f["ank_y"]/f["torso_h"] for f in frames])
    # fill nans
    nan = np.isnan(y)
    y[nan] = np.interp(np.flatnonzero(nan),
                      np.flatnonzero(~nan), y[~nan])
    ys = moving_average(y, smooth_w)
    # upward velocity = negative derivative * fps
    vel = -(ys[1:]-ys[:-1]) * fps
    mu, σ = vel.mean(), vel.std()
    thresh = mu + sigma_mul*σ
    peaks,_ = find_peaks(vel, height=thresh, distance=min_dist_frames)
    if len(peaks)==0:
        return None, vel, thresh
    # first spike → liftoff frame = peak+1 (because vel is one shorter)
    return int(peaks[0]+1), vel, thresh

def detect_max_kneelift(frames, smooth_w=7, min_prom=0.0, min_dist=5):
    # knee_y_norm time series
    y = np.array([f["knee_y"]/f["torso_h"] for f in frames])
    nan = np.isnan(y)
    y[nan] = np.interp(np.flatnonzero(nan),
                      np.flatnonzero(~nan), y[~nan])
    ys = moving_average(y, smooth_w)
    # global peak or the highest local peak
    peaks,_ = find_peaks(ys, distance=min_dist, prominence=min_prom)
    if len(peaks)==0:
        return None, ys
    # pick the peak with max height
    best = peaks[np.argmax(ys[peaks])]
    return int(best), ys

def extract_time_series(video_path, skip):
    mp_pose = mp.solutions.pose
    pose   = mp_pose.Pose(static_image_mode=True)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    frames = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        if frame_idx % skip == 0:
            h, w, _ = frame.shape
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(img)
            if res.pose_landmarks:
                lm = res.pose_landmarks.landmark
                def px(p): return (lm[p].x*w, lm[p].y*h)
                # back‐ankle for liftoff, front‐knee for max knee
                ank = px(mp_pose.PoseLandmark.RIGHT_ANKLE)
                kn  = px(mp_pose.PoseLandmark.RIGHT_KNEE)
                sh = px(mp_pose.PoseLandmark.RIGHT_SHOULDER)
                hp = px(mp_pose.PoseLandmark.RIGHT_HIP)
                torso_h = abs(sh[1] - hp[1]) or 1
                frames.append({
                    "idx": frame_idx,
                    "ank_y": ank[1],
                    "knee_y": kn[1],
                    "torso_h": torso_h
                })
            else:
                frames.append({
                    "idx": frame_idx,
                    "ank_y": np.nan,
                    "knee_y": np.nan,
                    "torso_h": np.nan
                })
        frame_idx += 1

    cap.release()
    return frames, fps

def main():
    p = argparse.ArgumentParser(
        description="Heuristic LiftOff / MaxKneeLift Detection"
    )
    p.add_argument("--video", required=True, help="Path to video file")
    p.add_argument("--skip",      type=int, default=1,
                   help="Process every Nth frame")
    p.add_argument("--lk_smooth", type=int, default=5,
                   help="Smoothing window for LiftOff velocity")
    p.add_argument("--lk_sigma_mul", type=float, default=1.5,
                   help="σ‐multiplier for liftoff threshold")
    p.add_argument("--lk_min_dist", type=int, default=5,
                   help="Min frames between velocity peaks")
    p.add_argument("--kl_smooth", type=int, default=7,
                   help="Smoothing window for knee height")
    p.add_argument("--kl_min_dist", type=int, default=5,
                   help="Min frames between knee peaks")
    args = p.parse_args()

    print("▶ Loading and extracting time series…")
    frames, fps = extract_time_series(args.video, args.skip)
    print(f"   Total sampled frames: {len(frames)}, FPS={fps:.1f}")

    print("\n▶ Detecting LiftOff…")
    lo_frame, vel, thresh = detect_liftoff(
        frames, fps,
        smooth_w=args.lk_smooth,
        sigma_mul=args.lk_sigma_mul,
        min_dist_frames=args.lk_min_dist
    )
    if lo_frame is None:
        print("  ❌ No lift-off peak found with σ_mul=", args.lk_sigma_mul)
    else:
        print(f"  ✅ LiftOff @ frame {frames[lo_frame]['idx']} "
              f"(vel_peak={vel[lo_frame-1]:.3f} ≥ {thresh:.3f})")

    print("\n▶ Detecting MaxKneeLift…")
    mk_frame, ys = detect_max_kneelift(
        frames,
        smooth_w=args.kl_smooth,
        min_dist=args.kl_min_dist
    )
    if mk_frame is None:
        print("  ❌ No knee‐lift peak found")
    else:
        print(f"  ✅ MaxKneeLift @ frame {frames[mk_frame]['idx']} "
              f"(height={ys[mk_frame]:.3f})")

if __name__=="__main__":
    main()
