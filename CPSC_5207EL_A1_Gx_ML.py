# Parts of the code below was generated with GitHub Copilot Agents using Claude Sonnet 4.5
# Reference to the conversation: gx_yolo_prompt.json

import cv2
from ultralytics import YOLO
import numpy as np
import time


def estimate_distance(pixel_size, real_size_cm=15.0, frame_width=640):
    """
    Estimates distance to traffic sign without camera calibration.

    Args:
        pixel_size: Average width/height of the bounding box in pixels
        real_size_cm: Real physical size of the traffic sign in cm (default: 15 cm)
        frame_width: Frame width in pixels (default: 640)

    Returns:
        float: Estimated distance in centimeters

    Note:
        This uses an approximate focal length based on typical webcam FOV (~60-70 degrees).
        For 640x480 resolution, focal length ≈ frame_width / (2 * tan(FOV/2))
        Assuming 60° horizontal FOV: f ≈ 640 / (2 * tan(30°)) ≈ 554 pixels
    """
    # Approximate focal length for typical webcam (in pixels)
    # This assumes ~60 degree horizontal field of view
    focal_length_px = frame_width / (2 * np.tan(np.radians(30)))

    # Distance = (focal_length * real_size) / pixel_size
    # Prevent division by zero
    if pixel_size > 0:
        distance_cm = (focal_length_px * real_size_cm) / pixel_size
        return distance_cm
    return None


# Load the fine-tuned YOLO model
MODEL_PATH = "traffic_sign_detection/yolov8n_finetuned/weights/best.pt"
model = YOLO(MODEL_PATH)

# Get class names from the model
class_names = model.names

# Initialize video capture (0 for default webcam)
cap = cv2.VideoCapture(0)

# Set video properties for better performance
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Starting live video detection...")
print("Press 'q' to quit")
print(f"Loaded model from: {MODEL_PATH}")
print(f"Detected classes: {class_names}")

# FPS calculation variables
prev_time = time.time()
fps_display = 0

while True:
    # Read frame from webcam
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    # Perform inference
    results = model(frame, conf=0.5, iou=0.45, verbose=False)

    # Process results
    for result in results:
        boxes = result.boxes

        # Draw bounding boxes and labels
        for box in boxes:
            # Get box coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Get confidence and class
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = class_names[class_id]

            # Calculate bounding box size for distance estimation
            box_width = x2 - x1
            box_height = y2 - y1
            box_size = box_width if box_width > box_height else box_height

            # Estimate distance to the traffic sign
            distance_cm = estimate_distance(box_size, real_size_cm=8.0, frame_width=640)

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Prepare label text with distance
            if distance_cm:
                label = f"{class_name}: {confidence:.2f} | {distance_cm:.1f}cm"
            else:
                label = f"{class_name}: {confidence:.2f}"

            # Get text size for background
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )

            # Draw background rectangle for text
            cv2.rectangle(
                frame,
                (x1, y1 - text_height - baseline - 5),
                (x1 + text_width, y1),
                (0, 255, 0),
                -1,
            )

            # Draw label text
            cv2.putText(
                frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2
            )

    # Calculate FPS
    current_time = time.time()
    fps_display = 1 / (current_time - prev_time)
    prev_time = current_time

    cv2.putText(
        frame,
        f"FPS: {fps_display:.1f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    # Display the frame
    cv2.imshow("Traffic Sign Detection - Live Video", frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
print("Video capture stopped.")
