"""Download a subset of the Seraphim Drone Detection Dataset and prepare it in YOLO format.

Dataset: https://huggingface.co/datasets/lgrzybowski/seraphim-drone-detection-dataset (CC BY 4.0)
83,483 images, single class (drone), 640x640, distributed as zipped batches.

Usage:
    python src/prepare_data.py --train-batches 1 --test-batches 1 --max-train 8000 --out data
"""
import argparse
import random
import shutil
import zipfile
from pathlib import Path

import yaml
from huggingface_hub import HfApi, hf_hub_download

REPO_ID = "lgrzybowski/seraphim-drone-detection-dataset"
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def batch_files(api: HfApi, split: str, n: int) -> list[tuple[str, str]]:
    files = api.list_repo_files(REPO_ID, repo_type="dataset")
    images = sorted(f for f in files if f.startswith(f"{split}/images/") and f.endswith(".zip"))
    labels = {Path(f).name: f for f in files if f.startswith(f"{split}/labels/") and f.endswith(".zip")}
    pairs = [(img, labels[Path(img).name]) for img in images if Path(img).name in labels]
    if not pairs:
        raise RuntimeError(f"No batches found for split '{split}'")
    return pairs[:n]


def download_and_extract(pairs, split: str, raw_dir: Path) -> None:
    for img_zip, lbl_zip in pairs:
        for remote, kind in [(img_zip, "images"), (lbl_zip, "labels")]:
            local = hf_hub_download(REPO_ID, remote, repo_type="dataset", cache_dir=raw_dir / ".cache")
            target = raw_dir / split / kind
            target.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(local) as z:
                for member in z.infolist():
                    if member.is_dir():
                        continue
                    name = Path(member.filename).name
                    with z.open(member) as src, open(target / name, "wb") as dst:
                        shutil.copyfileobj(src, dst)
            print(f"  downloaded: {remote}")


def pair_images(split_dir: Path) -> list[Path]:
    img_dir = split_dir / "images"
    files = [p for p in img_dir.iterdir() if p.is_file()] if img_dir.exists() else []
    images = sorted(p for p in files if p.suffix.lower() in IMG_EXT)
    if not images:
        sample = ", ".join(p.name for p in files[:5]) or "empty"
        raise RuntimeError(f"No images found in {img_dir} (files: {sample})")
    return images


def copy_split(images: list[Path], src_split: Path, dst_split: Path) -> None:
    (dst_split / "images").mkdir(parents=True, exist_ok=True)
    (dst_split / "labels").mkdir(parents=True, exist_ok=True)
    for img in images:
        shutil.copy2(img, dst_split / "images" / img.name)
        lbl = src_split / "labels" / (img.stem + ".txt")
        if lbl.exists():
            shutil.copy2(lbl, dst_split / "labels" / lbl.name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-batches", type=int, default=1)
    ap.add_argument("--test-batches", type=int, default=1)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--max-train", type=int, default=8000, help="max train+val images (0 = all)")
    ap.add_argument("--max-test", type=int, default=2000, help="max test images (0 = all)")
    ap.add_argument("--out", type=Path, default=Path("data"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    raw = args.out / "raw"
    api = HfApi()
    print("Downloading train batches...")
    download_and_extract(batch_files(api, "train", args.train_batches), "train", raw)
    print("Downloading test batches...")
    download_and_extract(batch_files(api, "test", args.test_batches), "test", raw)

    # validation split is carved from train; the test split stays untouched
    rng = random.Random(args.seed)
    train_imgs = pair_images(raw / "train")
    rng.shuffle(train_imgs)
    if args.max_train:
        train_imgs = train_imgs[: args.max_train]
    test_imgs = pair_images(raw / "test")
    rng.shuffle(test_imgs)
    if args.max_test:
        test_imgs = test_imgs[: args.max_test]
    n_val = int(len(train_imgs) * args.val_ratio)
    splits = {
        "train": (train_imgs[n_val:], raw / "train"),
        "val": (train_imgs[:n_val], raw / "train"),
        "test": (test_imgs, raw / "test"),
    }
    ds = args.out / "drone"
    for name, (imgs, src) in splits.items():
        copy_split(imgs, src, ds / name)
        n_lbl = sum(1 for _ in (ds / name / "labels").glob("*.txt"))
        print(f"{name:5s}: {len(imgs)} images, {n_lbl} label files")

    data_yaml = {
        "path": str(ds.resolve()),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "names": {0: "drone"},
    }
    with open(args.out / "drone.yaml", "w") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False)
    print(f"Done: {args.out / 'drone.yaml'}")
    shutil.rmtree(raw, ignore_errors=True)


if __name__ == "__main__":
    main()
