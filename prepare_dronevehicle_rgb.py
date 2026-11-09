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


def class_id(raw_name: str, xml_path: Path) -> int | None:
    if raw_name.strip() == "*":
        return None  # One genuinely ambiguous training label in the official release.
    key = "".join(ch for ch in raw_name.casefold() if ch.isalnum())
    aliases = {"car": 0, "freightcar": 1, "ferightcar": 1,
               "feright": 1, "truck": 2, "truvk": 2, "bus": 3, "van": 4}
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


def xml_points(obj: ET.Element, path: Path) -> tuple[np.ndarray | None, str]:
    box = obj.find("polygon")
    if box is None:
        box = obj.find("bndbox")
    if box is None:
        if obj.find("point") is not None:
            return None, "point"
        raise ValueError(f"Unknown annotation geometry in {path}")
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
            return points, "polygon"
    if box.tag == "bndbox":
        keys = ("xmin", "ymin", "xmax", "ymax")
        if all(box.findtext(key) is not None for key in keys):
            try:
                xmin, ymin, xmax, ymax = (float(box.findtext(key)) for key in keys)
            except ValueError as exc:
                raise ValueError(f"Non-numeric HBB coordinate in {path}") from exc
            if not np.isfinite([xmin, ymin, xmax, ymax]).all():
                raise ValueError(f"Non-finite HBB coordinate in {path}")
            points = np.array([[xmin, ymin], [xmax, ymin],
                               [xmax, ymax], [xmin, ymax]], dtype=np.float32)
            return points, "hbb"
    raise ValueError(f"Expected four OBB corners or HBB bounds in {path}")


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


def empty_audit() -> dict:
    return {
        "source_by_format": Counter(),
        "source_by_class_and_format": {name: Counter() for name in CLASSES},
        "retained_by_class": Counter(),
        "retained_by_class_and_format": {name: Counter() for name in CLASSES},
        "skipped_by_class_and_reason": {name: Counter() for name in CLASSES},
        "corrected_class_names": Counter(),
        "skipped_unresolved_names": Counter(),
        "examples": {},
    }


def merge_audit(total: dict, part: dict) -> None:
    for key in ("source_by_format", "retained_by_class", "corrected_class_names",
                "skipped_unresolved_names"):
        total[key].update(part[key])
    for key in ("source_by_class_and_format", "retained_by_class_and_format",
                "skipped_by_class_and_reason"):
        for name in CLASSES:
            total[key][name].update(part[key][name])
    for key, value in part["examples"].items():
        total["examples"].setdefault(key, value)


def convert_xml(path: Path) -> tuple[list[str], dict]:
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
    audit = empty_audit()
    for object_index, obj in enumerate(root.findall("object")):
        name = obj.findtext("name")
        if not name:
            raise ValueError(f"Object without a class name in {path}")
        points, geometry = xml_points(obj, path)
        audit["source_by_format"][geometry] += 1
        idx = class_id(name, path)
        example = {"xml": str(path), "object_index": object_index, "raw_name": name}
        if idx is None:
            audit["skipped_unresolved_names"][name] += 1
            audit["examples"].setdefault("unresolved_class", example)
            continue
        category = CLASSES[idx]
        audit["source_by_class_and_format"][category][geometry] += 1
        if name != category:
            audit["corrected_class_names"][name] += 1
            if name in ("feright", "truvk"):
                audit["examples"].setdefault(name, example)
        if geometry == "point":
            audit["skipped_by_class_and_reason"][category]["point_only"] += 1
            audit["examples"].setdefault("point_only", example)
            continue
        if geometry == "hbb":
            audit["examples"].setdefault("hbb", example)
        invalid_hbb = geometry == "hbb" and (
            points[1, 0] - points[0, 0] <= 1 or points[3, 1] - points[0, 1] <= 1
        )
        if invalid_hbb or cv2.contourArea(cv2.convexHull(points)) <= 1:
            audit["skipped_by_class_and_reason"][category]["degenerate"] += 1
            audit["examples"].setdefault("degenerate", example)
            continue
        width, height = OUTPUT_SIZE
        shifted = points - np.array([BORDER, BORDER], dtype=np.float32)
        if ((shifted[:, 0] < 0) | (shifted[:, 0] > width) |
                (shifted[:, 1] < 0) | (shifted[:, 1] > height)).any():
            audit["examples"].setdefault("crop_boundary", example)
        obb = clipped_obb(points)
        if obb is None:
            audit["skipped_by_class_and_reason"][category]["outside_crop"] += 1
            continue
        coords = " ".join(f"{float(v):.8f}" for v in obb.reshape(-1))
        lines.append(f"{idx} {coords}")
        audit["retained_by_class"][category] += 1
        audit["retained_by_class_and_format"][category][geometry] += 1
    return lines, audit


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
            split_audit = empty_audit()
            for image_index, (stem, image_path) in enumerate(sorted(images.items()), 1):
                image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if image is None or (image.shape[1], image.shape[0]) != SOURCE_SIZE:
                    raise ValueError(f"Unreadable or wrong-size RGB image: {image_path}")
                lines, image_audit = convert_xml(labels[stem])
                cropped = image[BORDER:-BORDER, BORDER:-BORDER]
                output_image = image_out / f"{stem}.png"
                if not cv2.imwrite(str(output_image), cropped):
                    raise OSError(f"Could not write image: {output_image}")
                (label_out / f"{stem}.txt").write_text(
                    "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
                )
                merge_audit(split_audit, image_audit)
                if lines:
                    preview_candidates.append((split, stem))
                if image_index % 1000 == 0 or image_index == len(images):
                    print(f"{split}: {image_index}/{len(images)} RGB images converted", flush=True)
            summary["splits"][split] = {
                "images": len(images),
                "objects_by_class": dict(split_audit["retained_by_class"]),
                "source_by_format": dict(split_audit["source_by_format"]),
                "source_by_class_and_format": {
                    name: dict(split_audit["source_by_class_and_format"][name]) for name in CLASSES
                },
                "retained_by_class_and_format": {
                    name: dict(split_audit["retained_by_class_and_format"][name]) for name in CLASSES
                },
                "skipped_by_class_and_reason": {
                    name: dict(split_audit["skipped_by_class_and_reason"][name]) for name in CLASSES
                },
                "corrected_class_names": dict(split_audit["corrected_class_names"]),
                "skipped_unresolved_names": dict(split_audit["skipped_unresolved_names"]),
                "examples": split_audit["examples"],
            }

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
