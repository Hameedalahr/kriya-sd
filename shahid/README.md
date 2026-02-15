# YOLO Object Detection - Video Processing

**Author:** Shahid  
**Task:** Detect objects in a video using YOLOv8 and save frames when a person is detected

---

## 📋 Task Description

Detect specified objects in a video using the YOLO object detection model. The system:
- Processes video frame by frame
- Detects objects using pre-trained YOLOv8
- Saves frames when a person is detected
- Creates output video with bounding boxes and labels

---

## 🚀 Setup

### 1. Virtual Environment

```bash
# Already activated
(venv) PS D:\kriya-sd>
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**Key Libraries:**
- `ultralytics` (YOLOv8)
- `opencv-python` (Video processing)
- `numpy` (Array operations)
- `torch` (PyTorch backend)

---

## 📂 Project Structure

```
shahid/
├── object_detection.py    # Main detection script
├── README.md             # This file
├── requirements.txt      # Dependencies
├── input/                # Place input videos here
│   └── video.mp4
└── output/               # Output videos saved here
    └── detected_video.mp4

data/
└── image/
    └── processed/        # Frames with person detection saved here
        ├── person_frame_000001.jpg
        ├── person_frame_000045.jpg
        └── ...
```

---

## 🎯 Usage

### Step 1: Add Your Video

Place your input video in the `shahid/input/` folder:

```bash
shahid/input/video.mp4
```

### Step 2: Run Detection

```bash
cd D:\kriya-sd
python shahid/object_detection.py
```

### Step 3: View Results

**Output Video:** `shahid/output/detected_video.mp4`  
**Saved Frames:** `data/image/processed/`

---

## ⚙️ Configuration

Edit `object_detection.py` to customize:

```python
# File paths
INPUT_VIDEO = "shahid/input/video.mp4"
OUTPUT_VIDEO = "shahid/output/detected_video.mp4"
FRAMES_DIR = "data/image/processed"

# Model settings
MODEL_NAME = "yolo11n.pt"  # Options: yolo11n.pt, yolo11s.pt, yolo11m.pt
CONFIDENCE_THRESHOLD = 0.5  # Detection confidence threshold (0.0-1.0)

# Detection targets
TARGET_CLASSES = None  # None = all classes, [0] = person only
PERSON_CLASS_ID = 0    # COCO class ID for 'person'
```

---

## 🎨 Output Features

### 1. Output Video
- Green bounding boxes around detected objects
- Labels showing class name and confidence score
- Same resolution and FPS as input video

### 2. Saved Frames
- Frames saved only when a person is detected
- High-quality JPEG images
- Numbered sequentially: `person_frame_XXXXXX.jpg`

---

## 📊 COCO Object Classes

YOLOv8 can detect 80 classes from COCO dataset:

| ID | Class | ID | Class | ID | Class |
|----|-------|----|----|----|----|
| 0 | person | 1 | bicycle | 2 | car |
| 3 | motorcycle | 4 | airplane | 5 | bus |
| 6 | train | 7 | truck | 8 | boat |
| ... | ... | ... | ... | ... | ... |

[Full COCO class list](https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/)

---

## 🔍 Example Output

```
============================================================
YOLO Object Detection - Video Processing
============================================================
✓ Directories set up
Loading YOLO model: yolo11n.pt...
✓ Model loaded successfully

Video Info:
  Resolution: 1920x1080
  FPS: 30
  Total Frames: 300

Processing video...
  Progress: 30/300 frames (10.0%)
  Progress: 60/300 frames (20.0%)
  ...
  Progress: 300/300 frames (100.0%)

✓ Processing complete!
  Total frames processed: 300
  Frames with person detected: 45
  Saved frames: data/image/processed
  Output video: shahid/output/detected_video.mp4

============================================================
✓ Task completed successfully!
============================================================
```

---

## 🛠️ Troubleshooting

### Issue: Video file not found
**Solution:** Ensure video is at `shahid/input/video.mp4`

### Issue: Low FPS / Slow processing
**Solution:** Use smaller model (`yolo11n.pt`) or reduce video resolution

### Issue: Too many/few detections
**Solution:** Adjust `CONFIDENCE_THRESHOLD` (higher = fewer, lower = more)

---

## 📝 Implementation Details

1. **Model Loading:** Downloads YOLOv8 model on first run (~6MB for nano)
2. **Frame Processing:** Each frame is processed independently
3. **Person Detection:** Checks if class_id == 0 (person)
4. **Frame Saving:** Saves complete frame (not cropped) when person detected
5. **Video Encoding:** Uses MP4V codec for compatibility

---

## 🎓 Deliverables

- ✅ Python script for object detection (`object_detection.py`)
- ✅ Output video file with detected objects
- ✅ Saved frames when person is detected
- ✅ Documentation (this README)

---

## 📌 Notes

- First run downloads the YOLO model (~6MB)
- Processing speed depends on video resolution and hardware
- GPU acceleration available if CUDA-enabled PyTorch installed
- Frames are saved with high quality (OpenCV default JPEG quality)

---

## 🔗 References

- [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com/)
- [OpenCV Documentation](https://docs.opencv.org/)
- [COCO Dataset](https://cocodataset.org/)

---

**Date:** February 15, 2026  
**Repository:** kriya-sd  
**Branch:** feature/shahid-task1
