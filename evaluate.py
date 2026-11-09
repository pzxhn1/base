"""Run Ultralytics OBB validation on packed RGB/thermal data."""

import argparse
from pathlib import Path

from ultralytics import YOLO

# Import custom model before unpickling the checkpoint.
from dual_yolo import DualBackboneOBB  # noqa: F401


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    model = YOLO(str(Path(args.weights).resolve()), task="obb")
    metrics = model.val(data=str(Path(args.data).resolve()), imgsz=args.imgsz,
                        batch=args.batch, device=args.device, plots=False)
    print({"mAP50": metrics.box.map50, "mAP50_95": metrics.box.map})


if __name__ == "__main__":
    main()
