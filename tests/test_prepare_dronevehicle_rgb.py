"""Checks for DroneVehicle cropping, OBB conversion, and pairing failures."""

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import yaml

from prepare_dronevehicle_rgb import OUTPUT_SIZE, clipped_obb, convert_xml, prepare


def write_sample(root: Path, split: str, stem: str, *, with_xml: bool = True,
                 class_name: str = "freight car") -> None:
    image_dir = root / split / f"{split}img"
    label_dir = root / split / f"{split}label"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    image = np.full((712, 840, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (120, 130), (180, 160), (50, 100, 150), -1)
    assert cv2.imwrite(str(image_dir / f"{stem}.jpg"), image)
    if with_xml:
        xml = """<annotation><size><width>840</width><height>712</height></size>
<object><name>{class_name}</name><polygon>
<x1>120</x1><y1>130</y1><x2>180</x2><y2>130</y2>
<x3>180</x3><y3>160</y3><x4>120</x4><y4>160</y4>
</polygon></object></annotation>""".format(class_name=class_name)
        (label_dir / f"{stem}.xml").write_text(xml, encoding="utf-8")


class PrepareDroneVehicleRGBTests(unittest.TestCase):
    def test_crops_and_converts_all_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, out = root / "source", root / "output"
            for split in ("train", "val", "test"):
                write_sample(source, split, "00001",
                             class_name="feright_car" if split == "val" else "freight car")
            summary = prepare(source, out, expected_counts=None, preview_count=3)
            self.assertEqual(summary["splits"]["train"]["objects_by_class"], {"freight_car": 1})
            image = cv2.imread(str(out / "images/train/00001.png"))
            self.assertEqual((image.shape[1], image.shape[0]), OUTPUT_SIZE)
            label = (out / "labels/train/00001.txt").read_text(encoding="utf-8").split()
            self.assertEqual(label[0], "1")
            self.assertEqual(len(label), 9)
            self.assertTrue(all(0 <= float(x) <= 1 for x in label[1:]))
            self.assertAlmostEqual(min(float(x) for x in label[1::2]), 20 / 640, places=5)
            self.assertEqual(len(list((out / "previews").glob("*.jpg"))), 3)
            config = yaml.safe_load((out / "data.yaml").read_text(encoding="utf-8"))
            self.assertEqual(config["names"][1], "freight_car")
            self.assertEqual(config["test"], "images/test")

    def test_missing_rgb_label_fails_before_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, out = root / "source", root / "output"
            for split in ("train", "val", "test"):
                write_sample(source, split, "00001", with_xml=split != "val")
            with self.assertRaisesRegex(ValueError, "No matching files|basenames differ"):
                prepare(source, out, expected_counts=None, preview_count=0)
            self.assertFalse(out.exists())

    def test_boundary_box_is_clipped_into_image(self):
        points = np.array([[90, 110], [130, 110], [130, 140], [90, 140]], dtype=np.float32)
        box = clipped_obb(points)
        self.assertIsNotNone(box)
        self.assertTrue(np.isfinite(box).all())
        self.assertTrue(((box >= 0) & (box <= 1)).all())
        self.assertAlmostEqual(float(box[:, 0].min()), 0, places=6)
        self.assertAlmostEqual(float(box[:, 0].max()), 30 / 640, places=6)

    def test_mixed_geometry_and_known_annotation_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            xml = Path(directory) / "mixed.xml"
            xml.write_text("""<annotation><size><width>840</width><height>712</height></size>
<object><name>car</name><bndbox><xmin>120</xmin><ymin>130</ymin><xmax>180</xmax><ymax>160</ymax></bndbox></object>
<object><name>car</name><point><x>200</x><y>200</y></point></object>
<object><name>feright</name><polygon><x1>120</x1><y1>130</y1><x2>180</x2><y2>130</y2><x3>180</x3><y3>160</y3><x4>120</x4><y4>160</y4></polygon></object>
<object><name>truvk</name><polygon><x1>120</x1><y1>130</y1><x2>180</x2><y2>130</y2><x3>180</x3><y3>160</y3><x4>120</x4><y4>160</y4></polygon></object>
<object><name>*</name><polygon><x1>120</x1><y1>130</y1><x2>180</x2><y2>130</y2><x3>180</x3><y3>160</y3><x4>120</x4><y4>160</y4></polygon></object>
<object><name>car</name><bndbox><xmin>120</xmin><ymin>130</ymin><xmax>120</xmax><ymax>160</ymax></bndbox></object>
</annotation>""", encoding="utf-8")
            lines, audit = convert_xml(xml)
            self.assertEqual(len(lines), 3)
            self.assertEqual([line.split()[0] for line in lines], ["0", "1", "2"])
            self.assertEqual(audit["retained_by_class_and_format"]["car"]["hbb"], 1)
            self.assertEqual(audit["skipped_by_class_and_reason"]["car"]["point_only"], 1)
            self.assertEqual(audit["skipped_by_class_and_reason"]["car"]["degenerate"], 1)
            self.assertEqual(audit["corrected_class_names"]["feright"], 1)
            self.assertEqual(audit["corrected_class_names"]["truvk"], 1)
            self.assertEqual(audit["skipped_unresolved_names"]["*"], 1)


if __name__ == "__main__":
    unittest.main()
