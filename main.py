import cv2
import os
from ultralytics import YOLO

# -----------------------------
# PATHS
# -----------------------------

VIDEO_PATH = "data/video/raw/Theft_attempt_in_Perumbavoor_Kerala_caught_on_security_camera_theft_perumbavoor_kallan_144P.mp4"
OUTPUT_VIDEO = "output_detected.mp4"
SAVE_DIR = "data/image/processed"

os.makedirs(SAVE_DIR, exist_ok=True)

# -----------------------------
# LOAD MODEL
# -----------------------------

model = YOLO("yolov8n.pt")

# -----------------------------
# OPEN VIDEO
# -----------------------------

cap = cv2.VideoCapture(VIDEO_PATH)

fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

saved_count = 0

# -----------------------------
# PROCESS FRAMES
# -----------------------------

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]

    person_detected = False

    for box in results.boxes:
        cls = int(box.cls[0])
        label = model.names[cls]

        if label == "person":
            person_detected = True

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, "Person", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    if person_detected:
        save_path = os.path.join(SAVE_DIR, f"frame_{saved_count}.jpg")
        cv2.imwrite(save_path, frame)
        saved_count += 1

    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()

print("Processing complete.")
print("Frames saved:", saved_count)
