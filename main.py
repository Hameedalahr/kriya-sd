import cv2
import os
from ultralytics import YOLO

# Load YOLOv8 Model

model = YOLO("yolov8n.pt")

# Paths

video_path = "data/video/raw/Theft_attempt_in_Perumbavoor_Kerala_caught_on_security_camera_theft_perumbavoor_kallan_144p.mp4"

output_video_path = "data/video/processed/output_detected.mp4"
output_image_folder = "data/image/processed"

# Create output folders if they don't exist
os.makedirs(output_image_folder, exist_ok=True)
os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

# Open Video

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Cannot open video file.")
    exit()

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Create video writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

frame_count = 0
person_detected_count = 0

print("Processing video...\n")

# Frame Processing Loop

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    annotated_frame = frame.copy()
    person_in_this_frame = False

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            label = model.names[class_id]

            # Detect only PERSON
            if label == "person":
                person_detected_count += 1
                person_in_this_frame = True

                # Get bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Draw bounding box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Add label
                cv2.putText(
                    annotated_frame,
                    f"{label} {confidence:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )

    # Save frame only once if person detected
    if person_in_this_frame:
        image_path = os.path.join(
            output_image_folder,
            f"person_frame_{frame_count}.jpg"
        )
        cv2.imwrite(image_path, frame)

    out.write(annotated_frame)
    frame_count += 1

# Release Resources

cap.release()
out.release()
cv2.destroyAllWindows()

print("Processing completed!")
print("Total frames processed:", frame_count)
print("Total person detections:", person_detected_count)
print("Output video saved at:", output_video_path)
