import cv2
import os
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Input video path
video_path = "kriya-sd/data/video/raw/Theft_attempt_in_Perumbavoor_Kerala_caught_on_security_camera_theft_perumbavoor_kallan_144p.mp4"

# Output paths
output_video_path = "kriya-sd/data/video/processed/output_detected.mp4"
output_image_folder = "kriya-sd/data/image/processed"

os.makedirs(output_image_folder, exist_ok=True)

# Open video
cap = cv2.VideoCapture(video_path)

# Get video properties
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

frame_count = 0
person_detected_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    annotated_frame = frame.copy()

    for result in results:
        boxes = result.boxes
        for box in boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            label = model.names[class_id]

            # Only highlight PERSON
            if label == "person":
                person_detected_count += 1

                # Save frame when person detected
                image_path = os.path.join(
                    output_image_folder,
                    f"person_detected_{frame_count}.jpg"
                )
                cv2.imwrite(image_path, frame)

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    annotated_frame,
                    f"{label} {confidence:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )

    out.write(annotated_frame)
    frame_count += 1

cap.release()
out.release()
cv2.destroyAllWindows()

print("Processing completed!")
print(f"Total person detections: {person_detected_count}")
