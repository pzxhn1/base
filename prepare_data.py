"""Pack paired RGB/thermal images into four-channel TIFFs for Ultralytics."""

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def image_map(folder):
    items = {p.stem: p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES}
    if not items:
        raise ValueError(f"No images in {folder}")
    return items


def check_label(path, n_classes):
    for row, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 9:
            raise ValueError(f"{path}:{row}: expected class + 8 normalized OBB corners")
        values = [float(v) for v in fields]
        if not values[0].is_integer() or not 0 <= int(values[0]) < n_classes:
            raise ValueError(f"{path}:{row}: invalid class id")
        if not all(np.isfinite(v) and 0 <= v <= 1 for v in values[1:]):
            raise ValueError(f"{path}:{row}: corners must be finite and normalized")


def prepare(source: Path, out: Path, names: list[str]):
    if source.resolve() == out.resolve():
        raise ValueError("Source and output directories must differ")
    counts = {}
    for split in ("train", "val"):
        root = source / split
        rgb_files = image_map(root / "rgb")
        ir_files = image_map(root / "thermal")
        if rgb_files.keys() != ir_files.keys():
            raise ValueError(f"{split}: RGB/thermal filenames differ: "
                             f"RGB-only={sorted(rgb_files.keys()-ir_files.keys())[:5]}, "
                             f"thermal-only={sorted(ir_files.keys()-rgb_files.keys())[:5]}")
        dst_img = out / "images" / split
        dst_lab = out / "labels" / split
        dst_img.mkdir(parents=True, exist_ok=True)
        dst_lab.mkdir(parents=True, exist_ok=True)
        for stem, rgb_path in sorted(rgb_files.items()):
            rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
            ir = cv2.imread(str(ir_files[stem]), cv2.IMREAD_GRAYSCALE)
            if rgb is None or ir is None or rgb.shape[:2] != ir.shape:
                raise ValueError(f"Unreadable or unequal-size pair: {stem}")
            label = root / "labels" / f"{stem}.txt"
            if not label.exists():
                raise FileNotFoundError(f"Missing label: {label}")
            check_label(label, len(names))
            packed = np.dstack((rgb, ir)).astype(np.uint8)
            out_image = dst_img / f"{stem}.tiff"
            if not cv2.imwrite(str(out_image), packed):
                raise OSError(f"Could not write {out_image}")
            roundtrip = cv2.imread(str(out_image), cv2.IMREAD_UNCHANGED)
            if roundtrip is None or roundtrip.shape != packed.shape or not np.array_equal(roundtrip, packed):
                raise OSError(f"Four-channel TIFF roundtrip failed: {out_image}")
            shutil.copyfile(label, dst_lab / f"{stem}.txt")
        counts[split] = len(rgb_files)
    config = {"path": str(out.resolve()), "train": "images/train", "val": "images/val",
              "channels": 4, "names": dict(enumerate(names))}
    (out / "data.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return counts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--names", required=True, help="Comma-separated class names in label-id order")
    args = p.parse_args()
    names = [x.strip() for x in args.names.split(",")]
    if not names or any(not x for x in names):
        p.error("--names must list every class in label-id order")
    print(prepare(args.source, args.out, names))


if __name__ == "__main__":
    main()
