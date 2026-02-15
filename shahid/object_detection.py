"""
YOLO Object Detection on Video
This script detects objects in a video using YOLOv8 and saves frames when a person is detected.
"""

import cv2
import os
from ultralytics import YOLO
from pathlib import Path

# Configuration
INPUT_VIDEO = "videoplayback.mp4"  # Input video path
OUTPUT_VIDEO = "shahid/output/detected_video.mp4"  # Output video path
FRAMES_DIR = "data/image/processed"  # Directory to save frames with person detection
MODEL_NAME = "yolo11n.pt"  # YOLOv8 nano model (fastest, can use yolov8s.pt for better accuracy)
CONFIDENCE_THRESHOLD = 0.5  # Confidence threshold for detections
TARGET_CLASSES = None  # None = detect all classes, or specify list like [0] for person only

# COCO class names (YOLOv8 uses COCO dataset)
# Class 0 = 'person'
PERSON_CLASS_ID = 0


def setup_directories():
    """Create necessary directories if they don't exist"""
    os.makedirs(os.path.dirname(OUTPUT_VIDEO), exist_ok=True)
    os.makedirs(FRAMES_DIR, exist_ok=True)
    print(f"✓ Directories set up")


def load_model():
    """Load YOLOv8 model"""
    print(f"Loading YOLO model: {MODEL_NAME}...")
    model = YOLO(MODEL_NAME)
    print("✓ Model loaded successfully")
    return model


def process_video(model):
    """Process video frame by frame and detect objects"""
    
    # Open video file
    if not os.path.exists(INPUT_VIDEO):
        raise FileNotFoundError(f"Video file not found: {INPUT_VIDEO}")
    
    cap = cv2.VideoCapture(INPUT_VIDEO)
    
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {INPUT_VIDEO}")
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\nVideo Info:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Total Frames: {total_frames}")
    
    # Define video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))
    
    frame_count = 0
    person_frame_count = 0
    
    print(f"\nProcessing video...")
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        frame_count += 1
        
        # Perform object detection
        results = model(frame, conf=CONFIDENCE_THRESHOLD, classes=TARGET_CLASSES, verbose=False)
        
        # Check if person is detected
        person_detected = False
        
        # Get detection results
        for result in results:
            boxes = result.boxes
            
            # Draw bounding boxes
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                
                # Check if person is detected
                if class_id == PERSON_CLASS_ID:
                    person_detected = True
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw label
                label = f"{class_name}: {confidence:.2f}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                            (x1 + label_size[0], y1), (0, 255, 0), -1)
                cv2.putText(frame, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        # Save frame if person detected
        if person_detected:
            person_frame_count += 1
            frame_filename = os.path.join(FRAMES_DIR, f"person_frame_{frame_count:06d}.jpg")
            cv2.imwrite(frame_filename, frame)
        
        # Write frame to output video
        out.write(frame)
        
        # Print progress
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"  Progress: {frame_count}/{total_frames} frames ({progress:.1f}%)")
    
    # Release resources
    cap.release()
    out.release()
    
    print(f"\n✓ Processing complete!")
    print(f"  Total frames processed: {frame_count}")
    print(f"  Frames with person detected: {person_frame_count}")
    print(f"  Saved frames: {FRAMES_DIR}")
    print(f"  Output video: {OUTPUT_VIDEO}")


def main():
    """Main function"""
    print("=" * 60)
    print("YOLO Object Detection - Video Processing")
    print("=" * 60)
    
    try:
        # Setup
        setup_directories()
        
        # Load model
        model = load_model()
        
        # Process video
        process_video(model)
        
        print("\n" + "=" * 60)
        print("✓ Task completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
