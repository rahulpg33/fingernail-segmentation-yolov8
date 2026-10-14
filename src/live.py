"""Live fingernail tracking on a PC, with bounding boxes and masks.

Runs a custom-trained YOLOv8 segmentation model on the default webcam and
tracks each fingernail across frames with ByteTrack (thesis section 5.4.1.1).

Usage:
    python src/live.py --model models/Yolov8n.pt --source 0
"""
import argparse
import os

# Avoid OpenMP "duplicate library" runtime errors on some Windows setups.
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Live fingernail tracking (YOLOv8 + ByteTrack)")
    parser.add_argument("--model", default="models/Yolov8n.pt", help="Path to trained YOLOv8 segmentation weights")
    parser.add_argument("--source", default="0", help="Webcam index or video file path")
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source

    model = YOLO(args.model)
    print(model)

    # Track objects from the webcam and show the annotated video.
    # Labels are hidden because the only class is "fingernail".
    model.track(source=source, show=True, tracker="bytetrack.yaml", show_labels=False)


if __name__ == "__main__":
    main()
