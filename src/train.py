"""Train a YOLOv8 segmentation model on the fingernail dataset.

Local equivalent of the Colab notebooks in notebooks/ (thesis section 5.2):
transfer learning from COCO-pre-trained weights, 100 epochs.

Usage:
    python src/train.py --size n          # yolov8n-seg
    python src/train.py --size l --epochs 100
"""
import argparse

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 fingernail segmentation")
    parser.add_argument("--size", choices=["n", "s", "m", "l", "x"], default="n", help="YOLOv8 model size")
    parser.add_argument("--data", default="dataset/data.yaml", help="Path to data.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    model = YOLO(f"yolov8{args.size}-seg.pt")
    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz)
    model.val()


if __name__ == "__main__":
    main()
