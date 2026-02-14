from ultralytics import YOLO

# Load pretrained YOLOv8 model
model = YOLO("yolov8n.pt")

# Path to video file (update according to your structure)
video_path = "data/video/video.mp4"  # <-- make sure the video file name matches exactly

# Run object detection
results = model(video_path, show=True, save=True)

print("Object detection completed successfully!")