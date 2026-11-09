"""Train the paired RGB/thermal mean-fusion OBB baseline."""

import argparse
from pathlib import Path

from ultralytics import YOLO
from dual_yolo import DualOBBTrainer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="Prepared data.yaml (channels: 4)")
    p.add_argument("--model", default="yolo11n-obb.yaml", help="YOLO11n-OBB YAML or .pt")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default="cpu")
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--project", default="runs/baseline")
    args = p.parse_args()
    model = YOLO(args.model, task="obb")
    model.train(data=str(Path(args.data).resolve()), trainer=DualOBBTrainer,
                epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                device=args.device, workers=args.workers, project=args.project,
                seed=42, deterministic=True, plots=False)


if __name__ == "__main__":
    main()
