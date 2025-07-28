#!/usr/bin/env python3
# retrain_keyframe_classifier.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os

# 1) Paths to your artifacts
FEATURE_CSV    = "keyframe_features_dataset.csv"    # existing features + original labels
FEEDBACK_CSV   = "coach_feedback.csv"               # coach_feedback.csv has: video,frame,predicted_label,is_correct,true_label,notes
NEW_FEATURE_CSV= "features_with_feedback.csv"
MODEL_OUT      = "keyframe_classifier.joblib"

# 2) Load original features + labels
df_feats = pd.read_csv(FEATURE_CSV)

# 3) Pull in coach feedback and turn into corrected labels
df_fb = pd.read_csv(FEEDBACK_CSV)
# keep only the incorrect ones and their corrected true label
df_fb = df_fb[df_fb.is_correct==False][["video","frame","true_label"]]
df_fb.rename(columns={"video":"Video","frame":"Frame","true_label":"Keyframe_Type"}, inplace=True)

# 4) Replace original labels with coach’s true labels
df = df_feats.merge(df_fb, on=["Video","Frame"], how="left", suffixes=("","_fb"))
df["Keyframe_Type"] = df["Keyframe_Type_fb"].fillna(df["Keyframe_Type"])
df.drop(columns=["Keyframe_Type_fb"], inplace=True)

# 5) (Optional) save this merged feature file for record
df.to_csv(NEW_FEATURE_CSV, index=False)
print(f"[+] Merged features+feedback → {NEW_FEATURE_CSV!r} (n={len(df)})")

# 6) Prepare X, y
feature_cols = [
    'Elbow_Angle','Trunk_Lean_Angle','Stride_Length','Norm_Stride',
    'Humerus_Straightness','UpperArm_Compression','Elbow_Angular_Velocity',
    'rKnee_y','rAnkle_x','rAnkle_y','lAnkle_x','lAnkle_y'
]
X = df[feature_cols]
y = df["Keyframe_Type"]

# 7) Train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# 8) Fit a new RandomForest
clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=42,
    class_weight="balanced"
)
print("[*] Training RandomForest…")
clf.fit(X_train, y_train)

# 9) Evaluate
print("\n[*] Test set performance:")
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred, zero_division=0))

# 10) Save updated model
joblib.dump(clf, MODEL_OUT)
print(f"[+] New model saved to {MODEL_OUT!r}")
