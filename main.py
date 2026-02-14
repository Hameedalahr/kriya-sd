import cv2
import os
from ultralytics import YOLO

# 1. Load YOLO model
model = YOLO("yolov8n.pt")

# 2. Video path
video_path = "data/video/raw/Theft_attempt_in_Perumbavoor_Kerala_caught_on_security_camera_theft_perumbavoor_kallan_144P.mp4"

# 3. Output folder
frame_output_dir = "data/image/processed"
os.makedirs(frame_output_dir, exist_ok=True)

# 4. Read video
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("❌ Error: Cannot open input video")
    exit()

# 5. SAFE VIDEO WRITER SETU
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Windows safe
fps = cap.get(cv2.CAP_PROP_FPS)

# FORCE FPS if invalid
if fps == 0 or fps is None:
    fps = 25

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

output_video_path = "output_detected.avi"
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

if not out.isOpened():
    print("❌ Error: VideoWriter failed")
    exit()


# 6. Process frames

frame_count = 0
saved_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    results = model(frame, conf=0.5)

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]

            if label == "person":
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                cv2.rectangle(frame, (x1, y1), (x2, y2),
                              (0, 255, 0), 2)
                cv2.putText(frame, "Person",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 255, 0), 2)

                frame_name = f"person_{saved_count}.jpg"
                cv2.imwrite(os.path.join(frame_output_dir, frame_name), frame)
                saved_count += 1

    out.write(frame)
    cv2.imshow("YOLO Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break


# 7. Cleanup

cap.release()
out.release()
cv2.destroyAllWindows()

print("DONE")
print("Frames processed:", frame_count)
print("Frames saved:", saved_count)
print("Output video:", output_video_path)
