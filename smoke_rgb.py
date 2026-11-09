"""Run one synthetic RGB OBB epoch and a test evaluation on the selected device."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import yaml


def make_image(root: Path, split: str, index: int) -> None:
    image_dir = root / "images" / split
    label_dir = root / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    image = np.random.default_rng(index).integers(0, 20, (128, 128, 3), dtype=np.uint8)
    x, y = 35 + index * 3, 65
    cv2.rectangle(image, (x - 12, y - 8), (x + 12, y + 8), (180, 140, 100), -1)
    if not cv2.imwrite(str(image_dir / f"{index:03d}.png"), image):
        raise OSError("Could not write synthetic image")
    corners = [(x - 12, y - 8), (x + 12, y - 8),
               (x + 12, y + 8), (x - 12, y + 8)]
    coords = " ".join(f"{value / 128:.6f}" for point in corners for value in point)
    (label_dir / f"{index:03d}.txt").write_text(f"0 {coords}\n", encoding="utf-8")


def main() -> None:
    device = sys.argv[1] if len(sys.argv) > 1 else "0"
    repo = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory(prefix="rgb_obb_smoke_") as directory:
        root = Path(directory)
        for split, indices in (("train", range(4)), ("val", range(4, 6)),
                               ("test", range(6, 8))):
            for index in indices:
                make_image(root, split, index)
        config = {"path": str(root), "train": "images/train", "val": "images/val",
                  "test": "images/test", "names": {0: "vehicle"}}
        data = root / "data.yaml"
        data.write_text(yaml.safe_dump(config), encoding="utf-8")
        project = root / "runs"
        subprocess.run(
            [sys.executable, str(repo / "train_rgb.py"), "--data", str(data),
             "--model", "yolo11n-obb.pt", "--epochs", "1", "--imgsz", "128",
             "--batch", "2", "--device", device, "--project", str(project),
             "--name", "smoke"], cwd=repo, check=True,
        )
        weights = project / "smoke" / "weights" / "best.pt"
        report = root / "test_metrics.json"
        subprocess.run(
            [sys.executable, str(repo / "evaluate_rgb.py"), "--data", str(data),
             "--weights", str(weights), "--out", str(report), "--imgsz", "128",
             "--batch", "2", "--device", device], cwd=repo, check=True,
        )
        result = json.loads(report.read_text(encoding="utf-8"))
        assert result["split"] == "test" and result["overall"]["mAP50"] >= 0
        print("PASS: RGB OBB train/save/test-metrics on", device)


if __name__ == "__main__":
    main()
