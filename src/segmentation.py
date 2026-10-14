"""Live fingernail segmentation on a Raspberry Pi 4 (YOLOv8n).

Adapted from the PC script for the Ultralytics version installed on the Pi
(thesis section 5.5): frames are converted BGR -> RGB and the mask is resized
to the frame before it is painted green. Press "q" to quit.

Setup on the Pi:
    python3 -m venv yolov8-env
    source yolov8-env/bin/activate
    pip install torch torchvision
    pip install opencv-python numpy ultralytics

Usage:
    python segmentation.py --model Yolov8n.pt
"""
import argparse
import os

# Avoid duplicate-library runtime errors.
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import cv2
import numpy as np
from ultralytics import YOLO

GREEN = (0, 255, 0)


def apply_segmentation(model, frame):
    """Segment fingernails in one frame and paint the masks green."""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # OpenCV uses BGR by default
    results = model(frame_rgb, verbose=False)

    if results[0].masks is None:
        return frame  # no fingernail found in this frame

    segmented = frame.copy()
    for mask in results[0].masks.data:
        mask = mask.cpu().numpy()
        mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
        segmented[mask > 0.5] = GREEN
    return segmented


def main():
    parser = argparse.ArgumentParser(description="Raspberry Pi live fingernail segmentation")
    parser.add_argument("--model", default="Yolov8n.pt", help="Path to trained YOLOv8n segmentation weights")
    parser.add_argument("--source", type=int, default=0, help="Webcam index")
    args = parser.parse_args()

    model = YOLO(args.model)
    print(model)

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise SystemExit("Error: could not open webcam.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            cv2.imshow("Segmented Frame", apply_segmentation(model, frame))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
