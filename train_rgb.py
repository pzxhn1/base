"""Train the unmodified Ultralytics YOLO11n-OBB RGB baseline."""

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/dronevehicle_rgb/data.yaml"))
    parser.add_argument("--model", default="yolo11n-obb.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fraction", type=float, default=1.0,
                        help="Fraction of training images; values below 1 are for a GPU probe")
    parser.add_argument("--project", default="runs/rgb_obb")
    parser.add_argument("--name", default="seed42")
    args = parser.parse_args()

    data = args.data.resolve()
    if not data.is_file():
        parser.error(f"Dataset YAML does not exist: {data}")
    if not 0 < args.fraction <= 1:
        parser.error("--fraction must be in (0, 1]")
    project = Path(args.project).resolve()
    if (project / args.name).exists():
        parser.error(f"Run directory already exists: {project / args.name}; choose another --name")
    if args.device != "cpu":
        if not torch.cuda.is_available():
            parser.error("CUDA is unavailable")
        capability = torch.cuda.get_device_capability(0)
        if capability != (12, 0):
            print(f"GPU compute capability: {capability}")
        # A real kernel call catches builds that report CUDA but lack sm_120.
        try:
            result = torch.ones(1, device="cuda:0").square().item()
        except RuntimeError as exc:
            parser.error(f"GPU kernel check failed: {exc}")
        if result != 1.0:
            parser.error("GPU kernel check returned an unexpected value")
        print(f"GPU ready: {torch.cuda.get_device_name(0)}; torch={torch.__version__}")

    model = YOLO(args.model, task="obb")
    model.train(
        data=str(data), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        device=args.device, workers=args.workers, seed=args.seed,
        deterministic=True, patience=args.epochs, fraction=args.fraction,
        project=str(project), name=args.name, plots=True,
    )


if __name__ == "__main__":
    main()
