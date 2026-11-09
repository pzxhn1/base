"""Evaluate the RGB OBB baseline and save machine-readable test metrics."""

import argparse
import hashlib
import json
import platform
from pathlib import Path

import torch
import torchvision
import ultralytics
from ultralytics import YOLO


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path,
                        default=Path("runs/rgb_obb/seed42/weights/best.pt"))
    parser.add_argument("--data", type=Path,
                        default=Path("data/dronevehicle_rgb/data.yaml"))
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--out", type=Path,
                        default=Path("runs/rgb_obb/seed42/test_metrics.json"))
    args = parser.parse_args()

    weights, data = args.weights.resolve(), args.data.resolve()
    if not weights.is_file() or not data.is_file():
        parser.error(f"Weights or dataset YAML missing: {weights}, {data}")
    output = args.out.resolve()
    model = YOLO(str(weights), task="obb")
    metrics = model.val(data=str(data), split=args.split, imgsz=args.imgsz,
                        batch=args.batch, device=args.device, workers=0,
                        seed=42, deterministic=True, plots=True,
                        project=str(output.parent), name=f"{args.split}_ultralytics")
    report = {
        "split": args.split,
        "checkpoint": str(weights),
        "checkpoint_sha256": sha256_file(weights),
        "data_yaml": str(data),
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "overall": {
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
            "mAP50": float(metrics.box.map50),
            "mAP50_95": float(metrics.box.map),
        },
        "per_class": metrics.summary(),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "ultralytics": ultralytics.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=lambda x: x.item()),
                      encoding="utf-8")
    print(json.dumps(report["overall"], indent=2))
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
