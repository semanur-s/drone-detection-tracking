"""Export the model to ONNX and compare PyTorch vs ONNX Runtime latency on CPU.

Usage:
    python src/export_benchmark.py --weights $(cat results/best_weights.txt) --images data/drone/test/images
"""
import argparse
import json
import time
from pathlib import Path

from ultralytics import YOLO

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def bench(model: YOLO, images: list[str], imgsz: int, device: str, warmup: int = 5) -> float:
    """Mean latency per image in ms."""
    for p in images[:warmup]:
        model.predict(p, imgsz=imgsz, device=device, verbose=False)
    t0 = time.perf_counter()
    for p in images:
        model.predict(p, imgsz=imgsz, device=device, verbose=False)
    return (time.perf_counter() - t0) / len(images) * 1000


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", type=Path, default=Path("results"))
    args = ap.parse_args()
    args.out.mkdir(exist_ok=True)

    images = sorted(str(p) for p in args.images.iterdir() if p.suffix.lower() in IMG_EXT)[: args.n]
    pt_model = YOLO(args.weights)
    onnx_path = pt_model.export(format="onnx", imgsz=args.imgsz, simplify=True)
    onnx_model = YOLO(onnx_path, task="detect")

    rows = []
    for name, model in [("PyTorch", pt_model), ("ONNX Runtime", onnx_model)]:
        ms = bench(model, images, args.imgsz, device="cpu")
        rows.append({"format": name, "cihaz": "CPU", "ms_goruntu": round(ms, 1), "fps": round(1000 / ms, 1)})
    rows.append({"onnx_boyut_mb": round(Path(onnx_path).stat().st_size / 1e6, 1)})

    (args.out / "benchmark.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    print(json.dumps(rows, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
