from ultralytics import YOLO

# Load YOLOv8 model (small version, fast)
model = YOLO("yolov8n.pt")  # will download automatically if not present

# Video path
video_path = "data/video/raw/Theft_attempt_in_Perumbavoor_Kerala_caught_on_security_camera_theft_perumbavoor_kallan_144P.mp4"

# Run object detection
results = model(video_path, show=True, save=True)

print("Object detection completed successfully!")