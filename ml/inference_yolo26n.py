import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

_BASE = Path(__file__).parent.parent

PT_MODEL       = str(_BASE / "models" / "yolo26n" / "weights" / "best.pt")
OPENVINO_MODEL = str(_BASE / "models" / "yolo26n" / "weights" / "best_openvino_model")


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO26n live inference — webcam")
    parser.add_argument(
        "--model",
        choices=["pt", "openvino"],
        default="pt",
        help="Backend to use: 'pt' for PyTorch weights, 'openvino' for OpenVINO export",
    )
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--iou",  type=float, default=0.45, help="IoU threshold")
    return parser.parse_args()


def estimate_distance(pixel_size, real_size_cm=8.0, frame_width=640):
    focal_length_px = frame_width / (2 * np.tan(np.radians(30)))
    if pixel_size > 0:
        return (focal_length_px * real_size_cm) / pixel_size
    return None


def draw_detections(frame, results, class_names):
    closest = {"class_name": None, "distance_cm": 0.0}

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            confidence = float(box.conf[0])
            class_name = class_names[int(box.cls[0])]

            box_size = max(x2 - x1, y2 - y1)
            distance_cm = estimate_distance(box_size, frame_width=frame.shape[1])

            if distance_cm and (
                distance_cm < closest["distance_cm"] or closest["distance_cm"] == 0.0
            ):
                closest = {"class_name": class_name, "distance_cm": distance_cm}

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            label = (
                f"{class_name}: {confidence:.2f} | {distance_cm:.1f}cm"
                if distance_cm
                else f"{class_name}: {confidence:.2f}"
            )
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - th - baseline - 5), (x1 + tw, y1), (0, 255, 0), -1)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    return frame, closest


def main():
    args = parse_args()

    if args.model == "openvino":
        model_path = OPENVINO_MODEL
        backend_label = "OpenVINO"
    else:
        model_path = PT_MODEL
        backend_label = "PyTorch (.pt)"

    print(f"\nBackend  : {backend_label}")
    print(f"Model    : {model_path}")
    print(f"Conf     : {args.conf}  |  IoU: {args.iou}")

    if not Path(model_path).exists():
        print(f"\nModel not found at: {model_path}")
        if args.model == "openvino":
            print("Export it first with:")
            print("  from ultralytics import YOLO")
            print("  YOLO('models/yolo26n/weights/best.pt').export(format='openvino')")
        else:
            print("Train it first with:  python ml/train_yolo26n.py")
        return

    model = YOLO(model_path)
    class_names = model.names
    print(f"Classes  : {list(class_names.values())}\n")
    print("Press 'q' to quit | 's' to toggle backend label on screen")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Could not open webcam.")
        return

    prev_time = time.time()
    show_info = True
    window_title = f"YOLO26n Traffic Sign Detection [{backend_label}]"

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        results = model(frame, conf=args.conf, iou=args.iou, verbose=False)
        frame, closest = draw_detections(frame, results, class_names)

        now = time.time()
        fps = 1.0 / (now - prev_time)
        prev_time = now

        if show_info:
            # FPS + backend overlay
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Backend: {backend_label}", (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 2)

        if closest["class_name"]:
            cv2.putText(frame, f"Navigation: {closest['class_name']}",
                        (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow(window_title, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            show_info = not show_info

    cap.release()
    cv2.destroyAllWindows()
    print("Stopped.")


if __name__ == "__main__":
    main()
