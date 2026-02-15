"""
Quick Test Script - YOLO Object Detection
Test the setup with webcam or a simple generated video
"""

import cv2
import numpy as np
from ultralytics import YOLO
import os


def test_webcam():
    """Test YOLO with webcam (press 'q' to quit)"""
    print("Testing YOLO with webcam...")
    print("Press 'q' to quit")
    
    # Load model
    model = YOLO("yolo11n.pt")
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Cannot open webcam")
        return
    
    frame_count = 0
    person_count = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        frame_count += 1
        
        # Run detection
        results = model(frame, conf=0.5, verbose=False)
        
        # Draw results
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                
                if class_id == 0:  # person
                    person_count += 1
                
                # Draw
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{class_name}: {confidence:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Show stats
        cv2.putText(frame, f"Frames: {frame_count}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Persons detected: {person_count}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display
        cv2.imshow('YOLO Test - Press Q to quit', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"✓ Test complete - {person_count} person detections in {frame_count} frames")


def create_sample_video():
    """Create a simple test video with moving shapes"""
    print("Creating sample test video...")
    
    output_path = "shahid/input/test_video.mp4"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Video settings
    width, height = 640, 480
    fps = 30
    duration = 5  # seconds
    total_frames = fps * duration
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for i in range(total_frames):
        # Create frame with moving circle
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Moving circle
        x = int((i / total_frames) * width)
        y = height // 2
        cv2.circle(frame, (x, y), 50, (0, 255, 0), -1)
        
        # Add text
        cv2.putText(frame, f"Frame {i+1}/{total_frames}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        out.write(frame)
    
    out.release()
    print(f"✓ Sample video created: {output_path}")
    print(f"  Resolution: {width}x{height}")
    print(f"  Duration: {duration}s, FPS: {fps}")
    return output_path


def test_model_loading():
    """Test if YOLO model loads correctly"""
    print("Testing YOLO model loading...")
    
    try:
        model = YOLO("yolo11n.pt")
        print("✓ Model loaded successfully")
        print(f"  Model type: {type(model)}")
        print(f"  Number of classes: {len(model.names)}")
        print(f"  Classes: {list(model.names.values())[:10]}...")
        return True
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False


def main():
    """Main test function"""
    print("=" * 60)
    print("YOLO Setup Test Script")
    print("=" * 60)
    
    # Test 1: Model loading
    print("\n[Test 1] Model Loading")
    if not test_model_loading():
        return
    
    # Test 2: Choose test type
    print("\n[Test 2] Choose a test:")
    print("  1. Test with webcam (requires camera)")
    print("  2. Create sample video")
    print("  3. Skip tests")
    
    choice = input("\nEnter choice (1/2/3): ").strip()
    
    if choice == "1":
        test_webcam()
    elif choice == "2":
        video_path = create_sample_video()
        print(f"\nYou can now run: python shahid/object_detection.py")
        print(f"Make sure to update INPUT_VIDEO to: {video_path}")
    elif choice == "3":
        print("Skipping tests")
    else:
        print("Invalid choice")
    
    print("\n" + "=" * 60)
    print("✓ Tests complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
