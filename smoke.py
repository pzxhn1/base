"""Generate tiny paired data and run one training/validation epoch on CPU."""

import tempfile
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from dual_yolo import DualOBBTrainer
from prepare_data import prepare


def make_sample(folder, index):
    rgb_dir = folder / "rgb"
    ir_dir = folder / "thermal"
    lab_dir = folder / "labels"
    for d in (rgb_dir, ir_dir, lab_dir):
        d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(index)
    rgb = rng.integers(0, 25, (128, 128, 3), dtype=np.uint8)
    ir = rng.integers(0, 25, (128, 128), dtype=np.uint8)
    cx, cy = 45 + index * 5, 65
    cv2.rectangle(rgb, (cx - 14, cy - 8), (cx + 14, cy + 8), (220, 160, 100), -1)
    cv2.rectangle(ir, (cx - 14, cy - 8), (cx + 14, cy + 8), 220, -1)
    cv2.imwrite(str(rgb_dir / f"{index}.png"), rgb)
    cv2.imwrite(str(ir_dir / f"{index}.png"), ir)
    corners = [(cx - 14, cy - 8), (cx + 14, cy - 8),
               (cx + 14, cy + 8), (cx - 14, cy + 8)]
    coords = " ".join(f"{v/128:.6f}" for point in corners for v in point)
    (lab_dir / f"{index}.txt").write_text(f"0 {coords}\n", encoding="utf-8")


def main():
    with tempfile.TemporaryDirectory(prefix="dual_yolo_smoke_") as temp:
        root = Path(temp)
        for i in range(4):
            make_sample(root / "raw" / "train", i)
        for i in range(2):
            make_sample(root / "raw" / "val", i + 10)
        prepare(root / "raw", root / "packed", ["vehicle"])
        model = YOLO("yolo11n-obb.yaml", task="obb")
        model.train(data=str(root / "packed" / "data.yaml"), trainer=DualOBBTrainer,
                    epochs=1, imgsz=128, batch=2, device="cpu", workers=0,
                    project=str(root / "runs"), name="smoke", exist_ok=True,
                    amp=False, plots=False, mosaic=0, mixup=0, seed=42)
        print("PASS: paired TIFF loading, one training epoch, OBB validation")


if __name__ == "__main__":
    main()
