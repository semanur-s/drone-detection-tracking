"""Fine-tune a YOLO detector on the drone dataset.

Usage:
    python src/train.py --data data/drone.yaml --model yolo11n.pt --epochs 50 --imgsz 640
"""
import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/drone.yaml")
    ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--device", default=None)
    ap.add_argument("--name", default="drone_yolo11n")
    args = ap.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        device=args.device,
        name=args.name,
        exist_ok=True,
        seed=42,
        mosaic=1.0,
        close_mosaic=10,
        plots=True,
    )

    best = Path(model.trainer.save_dir) / "weights" / "best.pt"
    Path("results").mkdir(exist_ok=True)
    Path("results/best_weights.txt").write_text(str(best))
    print(f"Best weights: {best}")


if __name__ == "__main__":
    main()
