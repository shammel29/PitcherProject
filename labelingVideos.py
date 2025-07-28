import pandas as pd

# Load keyframe info
keyframes_df = pd.read_csv("keyframes.csv")

# Store all pitch data here
all_pitches_data = []

for _, row in keyframes_df.iterrows():
    video_file = row['video']
    print(f"Processing: {video_file}")
    
    try:
        pitch_df = pd.read_csv(f"media_pipe_outputs/{video_file}")
    except FileNotFoundError:
        print(f"Missing file: {video_file}")
        continue

    features = {'video': video_file}

    for key in ['lift_off', 'circle_start', 'max_knee_lift', 'drag_start',
                'qtr_circle', 'circle_peak', 'three_qtr_circle', 'connection', 'release']:
        
        frame = int(row[key])
        frame_row = pitch_df[pitch_df['Frame'] == frame]

        if frame_row.empty:
            # Mark missing data
            features[f'elbow_angle_{key}'] = None
            features[f'trunk_lean_{key}'] = None
            features[f'stride_length_{key}'] = None
            features[f'wrist_hip_dist_{key}'] = None
        else:
            data = frame_row.iloc[0]
            features[f'elbow_angle_{key}'] = data['Elbow_Angle']
            features[f'trunk_lean_{key}'] = data['Trunk_Lean_Angle']
            features[f'stride_length_{key}'] = data['Stride_Length']
            features[f'wrist_hip_dist_{key}'] = data['Wrist_Hip_Distance']
    
    all_pitches_data.append(features)

# Convert to final dataframe
final_df = pd.DataFrame(all_pitches_data)
final_df.to_csv("pitch_keyframe_features.csv", index=False)
print("Done: pitch_keyframe_features.csv created")

labels_df = pd.read_csv("flaw_labels.csv")
features_df = pd.read_csv("pitch_keyframe_features.csv")

combined = pd.merge(features_df, labels_df, on="video")
combined.to_csv("final_training_data.csv", index=False)

