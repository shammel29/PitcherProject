import cv2, json, os
from batch_preprocess import render_overlay

# 1) Load events
outdir   = "demo_preproc"
video_in = "smallVideoSet/AudryeDumlao.mp4"
events   = json.load(open(os.path.join(outdir,"events.json")))

# 2) Make a set of event frames for quick lookup
event_frames = { ev["frame"]: ev["label"] for ev in events }

# 3) Open video & prepare writer
cap = cv2.VideoCapture(video_in)
fps    = cap.get(cv2.CAP_PROP_FPS)
w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out    = cv2.VideoWriter("demo_annotated.mp4", fourcc, fps, (w,h))

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 4) Draw skeleton on every frame
    ov = render_overlay(frame)

    # 5) If this frame is a key event, draw a colored banner
    if frame_idx in event_frames:
        label = event_frames[frame_idx]
        color = (0,255,0) if label=="Release" else (0,0,255)
        cv2.rectangle(ov, (0,0), (w,40), color, -1)
        cv2.putText(ov, f"{label} @ {frame_idx}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    out.write(ov)
    frame_idx += 1

cap.release()
out.release()
print("demo_annotated.mp4 ready!")
