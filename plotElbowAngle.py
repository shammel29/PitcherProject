import csv
import matplotlib.pyplot as plt

frames = []
angles = []

# Read the CSV data
with open("pitch_data.csv", newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        try:
            frame = int(row["Frame"])
            angle = float(row["Elbow_Angle"])
            frames.append(frame)
            angles.append(angle)
        except ValueError:
            # Skip rows with "NA" or bad data
            continue

# Plot the data
plt.figure(figsize=(10, 5))
plt.plot(frames, angles, marker='o', linestyle='-', color='blue')
plt.title("Elbow Angle During Softball Pitch")
plt.xlabel("Frame")
plt.ylabel("Elbow Angle (degrees)")
plt.grid(True)
plt.tight_layout()
plt.show()
