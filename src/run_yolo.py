"""Live fingernail segmentation on a PC, showing green masks without bounding boxes.

Each webcam frame goes through model.predict(); every predicted mask is painted
green on a copy of the frame (thesis section 5.4.1.2). Press "q" to quit.

Usage:
    python src/run_yolo.py --model models/Yolov8n.pt --source 0
"""
import argparse
import os

# Avoid OpenMP "duplicate library" runtime errors on some Windows setups.
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import cv2
import numpy as np
from ultralytics import YOLO

GREEN = (0, 255, 0)


def apply_segmentation(model, frame):
    """Run segmentation on one frame and paint every fingernail mask green."""
    results = model.predict(source=frame, stream=False, verbose=False)
    segmented = frame.copy()
    for result in results:
        if result.masks is None:
            continue
        for mask in result.masks.data:
            mask = mask.cpu().numpy()
            # Masks come out at model resolution; resize them to the frame size.
            mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
            segmented[mask > 0.5] = GREEN
    return segmented


def main():
    parser = argparse.ArgumentParser(description="Live fingernail segmentation (green masks, no boxes)")
    parser.add_argument("--model", default="models/Yolov8n.pt", help="Path to trained YOLOv8 segmentation weights")
    parser.add_argument("--source", default="0", help="Webcam index or video file path")
    args = parser.parse_args()

    model = YOLO(args.model)
    print(model)

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video source {args.source}")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            cv2.imshow("Fingernail segmentation", apply_segmentation(model, frame))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
