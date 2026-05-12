# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CPSC-5366EL Autonomous Mobile Robotics — Assignment 1. Compares two real-time traffic sign detection approaches:
- **CV approach**: Classical computer vision using OpenCV AprilTag (36h11 family) detection
- **ML approach**: Fine-tuned YOLOv8n/YOLOv11n deep learning models

## Repository Layout

```
Assignment_1/
├── cv/                  # Classical CV (AprilTag) approach
│   ├── detect.py        # Live webcam detection
│   └── measure.py       # 60-second FPS benchmark → results/
├── ml/                  # Machine learning (YOLO) approach
│   ├── detect.py        # Live webcam detection
│   ├── measure.py       # 60-second FPS benchmark → results/
│   ├── train_yolov8n.py # Fine-tune YOLOv8n → models/yolov8n/
│   ├── train_yolov11n.py# Fine-tune YOLOv11n → models/yolov11n/
│   └── data_augmentation.py  # Resize raw_data/ → resized/
├── analysis/            # Visualization scripts
│   ├── fps_graph_generator.py   # Reads results/fps_*.csv
│   └── map50_graph_generator.py # Reads models/yolov8n/results.csv
├── models/              # Trained model weights & training artifacts
│   ├── yolov8n/         # Fine-tuned YOLOv8n (weights/best.pt, results.csv, plots)
│   └── yolov11n/        # Fine-tuned YOLOv11n
├── results/             # All generated output
│   ├── fps_measurement_cv.csv
│   ├── fps_measurement_ml.csv
│   ├── fps_comparison_graphs/
│   └── training_comparison_graphs/
├── prompts/             # AI-assisted generation conversation logs
│   ├── cv_prompt.json
│   └── yolo_prompt.json
└── dataset/             # (gitignored) training data
```

## Environment Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Running the Main Programs

```bash
# Live AprilTag detection (CV approach, webcam required)
python cv/detect.py

# Live traffic sign detection (ML approach, requires trained weights)
python ml/detect.py

# FPS measurement — records 60 seconds then auto-stops (press 'q' to quit early)
python cv/measure.py    # outputs results/fps_measurement_cv_<timestamp>.csv
python ml/measure.py    # outputs results/fps_measurement_ml_<timestamp>.csv
```

## Training

```bash
# Fine-tune YOLOv8n (requires dataset/data.yaml and dataset/ split dirs)
python ml/train_yolov8n.py

# Fine-tune YOLOv11n
python ml/train_yolov11n.py

# Data preprocessing — resizes raw_data/ images to 640x480 into resized/
python ml/data_augmentation.py
```

## Generating Graphs

```bash
# FPS comparison graphs (reads results/fps_measurement_cv.csv and _ml.csv)
python analysis/fps_graph_generator.py    # outputs to results/fps_comparison_graphs/

# Training metrics graphs (reads models/yolov8n/results.csv)
python analysis/map50_graph_generator.py  # outputs to results/training_comparison_graphs/
```

All scripts resolve paths relative to `Path(__file__).parent.parent` (the repo root), so they work regardless of what directory you run them from.

## Architecture

### Detection approaches
- **CV** (`cv/detect.py`, `cv/measure.py`): Detects AprilTag 36h11 markers via `cv2.aruco.ArucoDetector`. Each tag ID maps to a navigation action (NO_ENTRY, DEAD_END, RIGHT, LEFT, FORWARD, STOP for IDs 0–5). Distance is estimated from average edge length using a pinhole model assuming 60° FOV.
- **ML** (`ml/detect.py`, `ml/measure.py`): Runs fine-tuned YOLO inference (conf=0.5, iou=0.45) on each webcam frame. Reports the closest detected sign by bounding-box-based distance estimate. Model and `class_names` are loaded at module level (not inside `main`).

### Model weights paths
- Pretrained base models: `yolov8n.pt`, `yolo11n.pt` (gitignored, auto-downloaded by ultralytics)
- Fine-tuned YOLOv8n: `models/yolov8n/weights/best.pt`
- Fine-tuned YOLOv11n: `models/yolov11n/weights/best.pt`

### Dataset
Expected at `dataset/` (gitignored) with `dataset/data.yaml` describing `train/`, `val/`, `test/` splits. Training uses 640×480 images, batch 16, 100 epochs with early stopping (patience=50). All YOLO augmentations are disabled except mosaic to preserve sign appearance fidelity.

6 classes (nc=6), class indices 0–5:

| Index | Name |
|-------|------|
| 0 | dead_end |
| 1 | forward |
| 2 | left |
| 3 | no_entry |
| 4 | right |
| 5 | stop |

Dataset sourced from Roboflow (`autonomousmobilerobotics/traffic-sign-detection-es8zl`, version 7, BY-NC-SA 4.0).

### Data augmentation pipeline (`ml/data_augmentation.py`)
Three transformer classes exist: `ImageTransformer` (resize + letterbox pad to 640×480), `GeometricTransformer` (rotation, shear, translation, scale), `PhotometricTransformer` (brightness, jitter, HSV shift, grayscale, noise). Currently only the resize step runs; geometric/photometric augmentation code is commented out.

### Distance estimation
Both approaches use the same formula without camera calibration:
```
focal_length_px = frame_width / (2 * tan(30°))   # assumes 60° horizontal FOV
distance_cm = (focal_length_px * real_size_cm) / pixel_size
```
CV uses `real_size_cm=7.5` (AprilTag), ML uses `real_size_cm=8.0` (bounding box largest dimension).
