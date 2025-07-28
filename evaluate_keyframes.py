import cv2
import pandas as pd
import os

# Config – edit paths as needed
video_path   = "videos/LucyStathis.mp4"
gt_csv       = "keyframes_labeled.csv"
pred_csv     = "keyframes_clustered.csv"
out_dir      = "manual_inspect"
tol          = 2

# Prep output folder
os.makedirs(out_dir, exist_ok=True)

# Load CSVs
gt   = pd.read_csv(gt_csv)   # Video,Keyframe_Type,Frame
pred = pd.read_csv(pred_csv) # Frame,Label,Confidence

# For each ground‐truth event:
for idx, row in gt.iterrows():
    label    = row['Keyframe_Type']
    gt_frame = int(row['Frame'])
    
    # Find predictions within ±tol
    cands = pred[(pred.Frame >= gt_frame-tol) &
                 (pred.Frame <= gt_frame+tol) &
                 (pred.Label == label)]
    
    # If none found, show the 3 closest predictions overall
    if cands.empty:
        # compute distance to each pred
        pred['dist'] = abs(pred.Frame - gt_frame)
        cands = pred.nsmallest(3, 'dist').copy()
        note = f"no_match_within_{tol}"
    else:
        note = "match"

    # Open video once
    cap = cv2.VideoCapture(video_path)
    
    # Extract & save GT frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, gt_frame)
    ret, f_gt = cap.read()
    if ret:
        gt_fname = f"{idx:02d}_{label}_GT_{gt_frame}.png"
        cv2.imwrite(os.path.join(out_dir, gt_fname), f_gt)
    
    # Extract & save each candidate
    for _, p in cands.iterrows():
        pf = int(p.Frame)
        cap.set(cv2.CAP_PROP_POS_FRAMES, pf)
        ret, f_pred = cap.read()
        if ret:
            pred_fname = f"{idx:02d}_{label}_{note}_{pf}_{p.Confidence:.2f}.png"
            cv2.imwrite(os.path.join(out_dir, pred_fname), f_pred)
    
    cap.release()

print(f"Saved frames for manual inspection in folder: {out_dir}")
print("You should see for each GT: one GT image and 1–3 candidate images to compare.")
