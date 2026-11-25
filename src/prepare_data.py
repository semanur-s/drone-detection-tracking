"""Seraphim Drone Detection veri setinden bir alt küme indirir ve YOLO formatında hazırlar.

Veri seti: https://huggingface.co/datasets/lgrzybowski/seraphim-drone-detection-dataset
Lisans: CC BY 4.0 — 83.483 görüntü, tek sınıf (drone), 640x640, YOLO formatında.

Veri seti zip'lenmiş "batch"ler halinde tutuluyor. Tamamı ~9 GB olduğu için Colab'da
sadece ilk birkaç batch indirilir. Test seti ayrı tutulur, doğrulama seti eğitimden ayrılır.

Kullanım:
    python src/prepare_data.py --train-batches 3 --test-batches 1 --out data
"""
import argparse
import random
import shutil
import zipfile
from pathlib import Path

import yaml
from huggingface_hub import HfApi, hf_hub_download

REPO_ID = "lgrzybowski/seraphim-drone-detection-dataset"
IMG_EXT = {".jpg", ".jpeg", ".png"}


def batch_files(api: HfApi, split: str, n: int) -> list[tuple[str, str]]:
    """Bir split için ilk n batch'in (görüntü zip'i, etiket zip'i) yollarını döndürür."""
    files = api.list_repo_files(REPO_ID, repo_type="dataset")
    images = sorted(f for f in files if f.startswith(f"{split}/images/") and f.endswith(".zip"))
    labels = {Path(f).name: f for f in files if f.startswith(f"{split}/labels/") and f.endswith(".zip")}
    pairs = [(img, labels[Path(img).name]) for img in images if Path(img).name in labels]
    if not pairs:
        raise RuntimeError(f"{split} için batch bulunamadı. Veri setinin yapısı değişmiş olabilir.")
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
                    # zip içindeki klasör yapısını düzleştir: sadece dosya adını al
                    name = Path(member.filename).name
                    with z.open(member) as src, open(target / name, "wb") as dst:
                        shutil.copyfileobj(src, dst)
            print(f"  indirildi: {remote}")


def pair_images(split_dir: Path) -> list[Path]:
    """Etiketi olan görüntüleri döndürür (etiketsiz görüntü = arka plan, yine de dahil edilir)."""
    return sorted(p for p in (split_dir / "images").iterdir() if p.suffix.lower() in IMG_EXT)


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
    ap.add_argument("--train-batches", type=int, default=3)
    ap.add_argument("--test-batches", type=int, default=1)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--out", type=Path, default=Path("data"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    raw = args.out / "raw"
    api = HfApi()
    print("Train batch'leri indiriliyor...")
    download_and_extract(batch_files(api, "train", args.train_batches), "train", raw)
    print("Test batch'leri indiriliyor...")
    download_and_extract(batch_files(api, "test", args.test_batches), "test", raw)

    # Eğitim verisinden doğrulama seti ayır; test seti modele hiç gösterilmez
    train_imgs = pair_images(raw / "train")
    random.Random(args.seed).shuffle(train_imgs)
    n_val = int(len(train_imgs) * args.val_ratio)
    splits = {
        "train": (train_imgs[n_val:], raw / "train"),
        "val": (train_imgs[:n_val], raw / "train"),
        "test": (pair_images(raw / "test"), raw / "test"),
    }
    ds = args.out / "drone"
    for name, (imgs, src) in splits.items():
        copy_split(imgs, src, ds / name)
        print(f"{name:5s}: {len(imgs)} görüntü")

    data_yaml = {
        "path": str(ds.resolve()),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "names": {0: "drone"},
    }
    with open(args.out / "drone.yaml", "w") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False)
    print(f"Hazır: {args.out / 'drone.yaml'}")


if __name__ == "__main__":
    main()
