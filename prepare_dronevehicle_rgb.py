"""Convert the official DroneVehicle RGB split to Ultralytics OBB format.

The official release has 840x712 images with a 100-pixel white border. This
script uses RGB images and RGB XML annotations only; IR files are untouched.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import tempfile
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import yaml


CLASSES = ("car", "freight_car", "truck", "bus", "van")
SPLITS = ("train", "val", "test")
EXPECTED_COUNTS = {"train": 17990, "val": 1469, "test": 8980}
SOURCE_SIZE = (840, 712)  # width, height
BORDER = 100
OUTPUT_SIZE = (640, 512)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def class_id(raw_name: str, xml_path: Path) -> int:
    key = "".join(ch for ch in raw_name.casefold() if ch.isalnum())
    aliases = {"car": 0, "freightcar": 1, "ferightcar": 1,
               "truck": 2, "bus": 3, "van": 4}
    if key not in aliases:
        raise ValueError(f"Unknown class {raw_name!r} in {xml_path}")
    return aliases[key]


def files_by_stem(folder: Path, suffixes: set[str]) -> dict[str, Path]:
    if not folder.is_dir():
        raise FileNotFoundError(f"Missing directory: {folder}")
    result = {}
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.suffix.casefold() in suffixes:
            if path.stem in result:
                raise ValueError(f"Duplicate basename in {folder}: {path.stem}")
            result[path.stem] = path
    if not result:
        raise ValueError(f"No matching files in {folder}")
    return result


def source_files(source: Path, split: str) -> tuple[dict[str, Path], dict[str, Path]]:
    root = source / split
    images = files_by_stem(root / f"{split}img", IMAGE_SUFFIXES)
    labels = files_by_stem(root / f"{split}label", {".xml"})
    if images.keys() != labels.keys():
        missing_labels = sorted(images.keys() - labels.keys())[:10]
        missing_images = sorted(labels.keys() - images.keys())[:10]
        raise ValueError(
            f"{split}: RGB image/XML basenames differ; "
            f"missing XML={missing_labels}, missing image={missing_images}"
        )
    return images, labels


def xml_points(obj: ET.Element, path: Path) -> np.ndarray:
    box = obj.find("polygon")
    if box is None:
        box = obj.find("bndbox")
    if box is None:
        raise ValueError(f"Missing <polygon> or <bndbox> in {path}")
    # DroneVehicle uses four corner points. Accept either 1..4 or 0..3 tags.
    for indices in ((1, 2, 3, 4), (0, 1, 2, 3)):
        if all(box.findtext(f"x{i}") is not None and box.findtext(f"y{i}") is not None for i in indices):
            try:
                points = np.array(
                    [[float(box.findtext(f"x{i}")), float(box.findtext(f"y{i}"))] for i in indices],
                    dtype=np.float32,
                )
            except ValueError as exc:
                raise ValueError(f"Non-numeric OBB coordinate in {path}") from exc
            if not np.isfinite(points).all():
                raise ValueError(f"Non-finite OBB coordinate in {path}")
            return points
    raise ValueError(f"Expected four OBB corner pairs in {path}")


def clipped_obb(points: np.ndarray) -> np.ndarray | None:
    shifted = points - np.array([BORDER, BORDER], dtype=np.float32)
    hull = cv2.convexHull(shifted).reshape(-1, 2)
    if len(hull) < 3 or cv2.contourArea(hull) <= 1:
        return None
    width, height = OUTPUT_SIZE
    frame = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
    area, clipped = cv2.intersectConvexConvex(hull, frame)
    if clipped is None or area <= 1:
        return None
    rectangle = cv2.boxPoints(cv2.minAreaRect(clipped)).astype(np.float32)
    rectangle[:, 0] = np.clip(rectangle[:, 0], 0, width)
    rectangle[:, 1] = np.clip(rectangle[:, 1], 0, height)
    if cv2.contourArea(rectangle) <= 1:
        return None
    return rectangle / np.array([width, height], dtype=np.float32)


def convert_xml(path: Path) -> tuple[list[str], Counter, int]:
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as exc:
        raise ValueError(f"Unreadable XML: {path}") from exc
    size = root.find("size")
    if size is not None:
        try:
            declared = (int(size.findtext("width")), int(size.findtext("height")))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid XML size in {path}") from exc
        if declared != SOURCE_SIZE:
            raise ValueError(f"Unexpected XML size {declared} in {path}; expected {SOURCE_SIZE}")
    lines: list[str] = []
    counts: Counter = Counter()
    outside = 0
    for obj in root.findall("object"):
        name = obj.findtext("name")
        if not name:
            raise ValueError(f"Object without a class name in {path}")
        idx = class_id(name, path)
        points = xml_points(obj, path)
        if cv2.contourArea(cv2.convexHull(points)) <= 1:
            raise ValueError(f"Degenerate OBB in {path}")
        obb = clipped_obb(points)
        if obb is None:
            outside += 1
            continue
        coords = " ".join(f"{float(v):.8f}" for v in obb.reshape(-1))
        lines.append(f"{idx} {coords}")
        counts[CLASSES[idx]] += 1
    return lines, counts, outside


def draw_preview(image_path: Path, label_path: Path, destination: Path) -> None:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unreadable converted image: {image_path}")
    height, width = image.shape[:2]
    for line in label_path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        corners = np.array([float(x) for x in fields[1:]], dtype=np.float32).reshape(4, 2)
        corners *= np.array([width, height], dtype=np.float32)
        cv2.polylines(image, [np.rint(corners).astype(np.int32)], True, (0, 255, 0), 2)
        cv2.putText(image, CLASSES[int(fields[0])], tuple(np.rint(corners[0]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), image):
        raise OSError(f"Could not write preview: {destination}")


def prepare(source: Path, out: Path, *, expected_counts: dict[str, int] | None = EXPECTED_COUNTS,
            preview_count: int = 20, seed: int = 42) -> dict:
    source, out = source.resolve(), out.resolve()
    if source == out or source in out.parents or out in source.parents:
        raise ValueError("Source and output directories must be separate")
    if out.exists():
        raise FileExistsError(f"Output already exists: {out}")
    source_maps = {split: source_files(source, split) for split in SPLITS}
    if expected_counts is not None:
        for split, (images, _) in source_maps.items():
            if len(images) != expected_counts[split]:
                raise ValueError(
                    f"{split}: found {len(images)} RGB pairs; expected {expected_counts[split]} "
                    "from the official full split"
                )

    out.parent.mkdir(parents=True, exist_ok=True)
    summary = {"source": str(source), "output": str(out), "modality": "RGB",
               "classes": list(CLASSES), "border_pixels": BORDER, "splits": {}}
    preview_candidates: list[tuple[str, str]] = []
    with tempfile.TemporaryDirectory(prefix="dronevehicle_rgb_", dir=out.parent) as temp:
        stage = Path(temp) / "converted"
        for split in SPLITS:
            images, labels = source_maps[split]
            image_out = stage / "images" / split
            label_out = stage / "labels" / split
            image_out.mkdir(parents=True)
            label_out.mkdir(parents=True)
            split_counts: Counter = Counter()
            outside_total = 0
            for stem, image_path in sorted(images.items()):
                image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if image is None or (image.shape[1], image.shape[0]) != SOURCE_SIZE:
                    raise ValueError(f"Unreadable or wrong-size RGB image: {image_path}")
                lines, counts, outside = convert_xml(labels[stem])
                cropped = image[BORDER:-BORDER, BORDER:-BORDER]
                output_image = image_out / f"{stem}.png"
                if not cv2.imwrite(str(output_image), cropped):
                    raise OSError(f"Could not write image: {output_image}")
                (label_out / f"{stem}.txt").write_text(
                    "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
                )
                split_counts.update(counts)
                outside_total += outside
                if lines:
                    preview_candidates.append((split, stem))
            summary["splits"][split] = {"images": len(images),
                                        "objects_by_class": dict(split_counts),
                                        "objects_outside_crop": outside_total}

        if len(preview_candidates) < preview_count:
            raise ValueError(f"Only {len(preview_candidates)} labeled images; need {preview_count} previews")
        selected = random.Random(seed).sample(preview_candidates, preview_count)
        for split, stem in selected:
            draw_preview(stage / "images" / split / f"{stem}.png",
                         stage / "labels" / split / f"{stem}.txt",
                         stage / "previews" / f"{split}_{stem}.jpg")
        config = {"path": str(out), "train": "images/train", "val": "images/val",
                  "test": "images/test", "names": dict(enumerate(CLASSES))}
        (stage / "data.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        (stage / "manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        shutil.move(str(stage), str(out))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True,
                        help="Official DroneVehicle root containing train/, val/, test/")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--preview-count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-subset", action="store_true", help="Only for pipeline checks")
    args = parser.parse_args()
    if args.preview_count < 0:
        parser.error("--preview-count must be non-negative")
    summary = prepare(args.source, args.out,
                      expected_counts=None if args.allow_subset else EXPECTED_COUNTS,
                      preview_count=args.preview_count, seed=args.seed)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
