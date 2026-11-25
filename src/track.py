"""Detect and track drones in a video with YOLO + ByteTrack.

Outputs (in --out):
    tracked.mp4       annotated video (boxes, IDs, trails)
    tracks.csv        frame, track_id, x1, y1, x2, y2, conf
    track_stats.json  number of tracks, mean track length, mean FPS

Usage:
    python src/track.py --weights $(cat results/best_weights.txt) --source video.mp4
"""
import argparse
import csv
import json
import time
from collections import defaultdict, deque
from pathlib import Path

import cv2
from ultralytics import YOLO


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--tracker", default="configs/bytetrack_drone.yaml")
    ap.add_argument("--conf", type=float, default=0.1)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--trail", type=int, default=30)
    args = ap.parse_args()
    args.out.mkdir(exist_ok=True)

    model = YOLO(args.weights)
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {args.source}")
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(args.out / "tracked.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (w, h))

    trails: dict[int, deque] = defaultdict(lambda: deque(maxlen=args.trail))
    lengths: dict[int, int] = defaultdict(int)
    times = []
    frame_idx = 0

    with open(args.out / "tracks.csv", "w", newline="") as f:
        log = csv.writer(f)
        log.writerow(["frame", "track_id", "x1", "y1", "x2", "y2", "conf"])
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t0 = time.perf_counter()
            res = model.track(frame, persist=True, tracker=args.tracker, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
            times.append(time.perf_counter() - t0)

            boxes = res.boxes
            if boxes is not None and boxes.id is not None:
                for (x1, y1, x2, y2), tid, c in zip(boxes.xyxy.tolist(), boxes.id.int().tolist(), boxes.conf.tolist()):
                    log.writerow([frame_idx, tid, round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1), round(c, 3)])
                    lengths[tid] += 1
                    trails[tid].append((int((x1 + x2) / 2), int((y1 + y2) / 2)))
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 200, 255), 2)
                    cv2.putText(frame, f"ID {tid} {c:.2f}", (int(x1), int(y1) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 2)
                    pts = list(trails[tid])
                    for a, b in zip(pts, pts[1:]):
                        cv2.line(frame, a, b, (255, 120, 0), 2)
            fps_now = 1 / times[-1] if times[-1] > 0 else 0
            cv2.putText(frame, f"FPS: {fps_now:.1f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            writer.write(frame)
            frame_idx += 1

    cap.release()
    writer.release()

    steady = times[5:] if len(times) > 10 else times  # skip warm-up frames
    stats = {
        "kare_sayisi": frame_idx,
        "iz_sayisi": len(lengths),
        "ortalama_iz_uzunlugu_kare": round(sum(lengths.values()) / len(lengths), 1) if lengths else 0,
        "ortalama_fps": round(len(steady) / sum(steady), 1) if steady else 0,
    }
    (args.out / "track_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
