"""Evaluate a trained model on the test split.

Reports standard detection metrics (P, R, mAP@0.5, mAP@0.5:0.95) and recall
per object size (COCO size bins: small < 32², medium 32²–96², large > 96² px).

Usage:
    python src/evaluate.py --weights $(cat results/best_weights.txt) --data data/drone.yaml
"""
import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from ultralytics import YOLO

SIZE_BUCKETS = {"kucuk (<32px)": (0, 32**2), "orta (32-96px)": (32**2, 96**2), "buyuk (>96px)": (96**2, float("inf"))}
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if len(boxes) == 0:
        return np.zeros(0)
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (box[2] - box[0]) * (box[3] - box[1])
    area_b = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / (area_a + area_b - inter + 1e-9)


def read_gt(label_path: Path, w: int, h: int) -> np.ndarray:
    """YOLO label (cls cx cy bw bh, normalized) -> pixel xyxy."""
    if not label_path.exists():
        return np.zeros((0, 4))
    rows = [list(map(float, l.split()[1:5])) for l in label_path.read_text().splitlines() if l.strip()]
    if not rows:
        return np.zeros((0, 4))
    a = np.array(rows)
    cx, cy, bw, bh = a[:, 0] * w, a[:, 1] * h, a[:, 2] * w, a[:, 3] * h
    return np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)


def size_recall(model: YOLO, test_images: Path, conf: float, iou_thr: float = 0.5) -> dict:
    found = {k: 0 for k in SIZE_BUCKETS}
    total = {k: 0 for k in SIZE_BUCKETS}
    images = sorted(p for p in test_images.iterdir() if p.suffix.lower() in IMG_EXT)
    # one image at a time: passing a list makes it a single batch and exhausts GPU memory
    for res in (r for p in images for r in model.predict(str(p), conf=conf, verbose=False)):
        h, w = res.orig_shape
        img_path = Path(res.path)
        gt = read_gt(img_path.parent.parent / "labels" / (img_path.stem + ".txt"), w, h)
        pred = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else np.zeros((0, 4))
        used = np.zeros(len(pred), dtype=bool)
        for g in gt:
            area = (g[2] - g[0]) * (g[3] - g[1])
            bucket = next(k for k, (lo, hi) in SIZE_BUCKETS.items() if lo <= area < hi)
            total[bucket] += 1
            ious = iou(g, pred)
            ious[used] = 0
            if len(ious) and ious.max() >= iou_thr:
                used[ious.argmax()] = True
                found[bucket] += 1
    return {k: {"adet": total[k], "recall": (found[k] / total[k]) if total[k] else None} for k in SIZE_BUCKETS}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", default="data/drone.yaml")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--out", type=Path, default=Path("results"))
    args = ap.parse_args()
    args.out.mkdir(exist_ok=True)

    model = YOLO(args.weights)
    m = model.val(data=args.data, split="test", imgsz=args.imgsz, plots=True, name="test_eval", exist_ok=True)
    metrics = {
        "precision": float(m.box.mp),
        "recall": float(m.box.mr),
        "mAP50": float(m.box.map50),
        "mAP50_95": float(m.box.map),
    }

    cfg = yaml.safe_load(open(args.data))
    metrics["boyuta_gore"] = size_recall(model, Path(cfg["path"]) / cfg["test"], args.conf)

    (args.out / "test_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
