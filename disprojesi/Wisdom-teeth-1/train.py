import torch
import torchvision
from ultralytics import YOLO

def main():
    model = YOLO("yolov8s.pt")

    results = model.train(
        data="data.yaml",
        epochs=43,
        imgsz=640,
        batch=16,
        name="disprojesi",
        project="trained_models",
        device=0,
        val=True,
        verbose=True
    )

if __name__ == "__main__":
    main()

