import pandas as pd

# Load your clustered predictions
pred = pd.read_csv('keyframes_clustered.csv')  # columns: Frame, Label, Confidence

# Video FPS—use the same value you ran detection with
fps = 30.0

# Compute time (in seconds) for each frame
pred['Time_sec'] = pred['Frame'] / fps

# Save with a nice human-readable format
pred.to_csv('keyframes_with_time.csv', index=False, float_format="%.3f")

print(pred[['Frame','Time_sec','Label','Confidence']])
