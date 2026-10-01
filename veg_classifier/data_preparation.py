"""
data_preparation.py
--------------------
Two ways to get data onto disk in the ImageFolder layout the rest of the
pipeline expects:  data/{train,val,test}/<class_name>/<image>.jpg

1. prepare_food101_subset()  -> REAL photos. Downloads the full Food-101
   dataset via torchvision (~5 GB, needs normal internet access) and copies
   out only the 20 vegetarian classes defined in config.py, using Food-101's
   official train/test split plus a stratified val split carved out of train.

2. generate_synthetic_dataset() -> Procedurally generated placeholder images
   (distinct colour palette + shape + texture per class). This is what lets
   the ENTIRE pipeline (loading -> preprocessing -> training -> evaluation
   -> demo) be executed and verified end-to-end in environments with no
   internet access, e.g. to sanity check the code before running it on the
   real dataset. Swap this call for #1 (or your own folder of photos) to
   train the model you actually submit.
"""

import os
import shutil
import random

from PIL import Image, ImageDraw, ImageFilter
import numpy as np

from . import config


# ---------------------------------------------------------------------------
# 1. REAL DATASET: Food-101 vegetarian subset
# ---------------------------------------------------------------------------
def prepare_food101_subset(root=None, val_fraction=0.15, seed=config.SEED):
    """
    Downloads Food-101 (torchvision) and materialises an ImageFolder-style
    train/val/test split containing ONLY the vegetarian classes in
    config.MEAL_CATEGORY_MAP. Requires unrestricted internet access.
    """
    import torchvision

    root = root or os.path.join(config.DATA_DIR, "_food101_raw")
    os.makedirs(root, exist_ok=True)

    print("Downloading Food-101 (first run only, ~5GB)...")
    train_full = torchvision.datasets.Food101(root=root, split="train", download=True)
    test_full = torchvision.datasets.Food101(root=root, split="test", download=True)

    rng = random.Random(seed)

    def _copy_split(dataset, split_name, holdout_val):
        # dataset._image_files / dataset._labels are populated by torchvision;
        # dataset.classes gives the label names.
        class_names = dataset.classes
        by_class = {}
        for path, label_idx in zip(dataset._image_files, dataset._labels):
            cname = class_names[label_idx]
            if cname not in config.MEAL_CATEGORY_MAP:
                continue
            by_class.setdefault(cname, []).append(path)

        for cname, paths in by_class.items():
            rng.shuffle(paths)
            if holdout_val:
                n_val = int(len(paths) * val_fraction)
                val_paths, train_paths = paths[:n_val], paths[n_val:]
                _copy_files(train_paths, os.path.join(config.TRAIN_DIR, cname))
                _copy_files(val_paths, os.path.join(config.VAL_DIR, cname))
            else:
                _copy_files(paths, os.path.join(config.TEST_DIR, cname))

    def _copy_files(paths, dest_dir):
        os.makedirs(dest_dir, exist_ok=True)
        for p in paths:
            shutil.copy(str(p), dest_dir)

    _copy_split(train_full, "train", holdout_val=True)
    _copy_split(test_full, "test", holdout_val=False)
    print(f"Vegetarian subset ready under {config.DATA_DIR}")


# ---------------------------------------------------------------------------
# 2. SYNTHETIC OFFLINE DATASET (pipeline smoke-test / no-internet demo)
# ---------------------------------------------------------------------------
def _class_visual_signature(class_name):
    """Deterministic colour palette + shape family per class, so the
    classification task is learnable (not just noise) but still clearly a
    stand-in for real photography."""
    rng = random.Random(class_name)
    base_color = tuple(rng.randint(40, 215) for _ in range(3))
    accent_color = tuple(rng.randint(40, 215) for _ in range(3))
    shape = rng.choice(["circle", "square", "triangle", "stripes"])
    return base_color, accent_color, shape


def _make_synthetic_image(class_name, size=256):
    base_color, accent_color, shape = _class_visual_signature(class_name)
    img = Image.new("RGB", (size, size), base_color)
    draw = ImageDraw.Draw(img)

    # per-sample jitter so images of the same class aren't identical
    jitter = lambda c: tuple(
        max(0, min(255, v + random.randint(-25, 25))) for v in c
    )
    accent = jitter(accent_color)

    if shape == "circle":
        r = random.randint(size // 4, size // 2)
        cx, cy = random.randint(r, size - r), random.randint(r, size - r)
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=accent)
    elif shape == "square":
        s = random.randint(size // 3, int(size * 0.7))
        x0, y0 = random.randint(0, size - s), random.randint(0, size - s)
        draw.rectangle([x0, y0, x0 + s, y0 + s], fill=accent)
    elif shape == "triangle":
        pts = [(random.randint(0, size), random.randint(0, size)) for _ in range(3)]
        draw.polygon(pts, fill=accent)
    else:  # stripes
        stripe_w = size // random.randint(4, 8)
        for x in range(0, size, stripe_w * 2):
            draw.rectangle([x, 0, x + stripe_w, size], fill=accent)

    img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 2.0)))
    # sensor-noise-like jitter
    arr = np.array(img).astype(np.int16)
    arr += np.random.randint(-12, 12, arr.shape, dtype=np.int16)
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def generate_synthetic_dataset(n_train=40, n_val=10, n_test=10, seed=config.SEED):
    """Creates data/{train,val,test}/<class>/img_*.jpg with procedurally
    generated placeholder images for every class in config.CLASSES."""
    random.seed(seed)
    np.random.seed(seed)

    for split_dir, n in [
        (config.TRAIN_DIR, n_train),
        (config.VAL_DIR, n_val),
        (config.TEST_DIR, n_test),
    ]:
        for cname in config.CLASSES:
            out_dir = os.path.join(split_dir, cname)
            os.makedirs(out_dir, exist_ok=True)
            for i in range(n):
                img = _make_synthetic_image(cname)
                img.save(os.path.join(out_dir, f"{cname}_{i:03d}.jpg"), quality=90)

    total = len(config.CLASSES) * (n_train + n_val + n_test)
    print(
        f"Synthetic dataset ready: {len(config.CLASSES)} classes x "
        f"({n_train}+{n_val}+{n_test}) images = {total} images total, "
        f"under {config.DATA_DIR}"
    )


if __name__ == "__main__":
    generate_synthetic_dataset()
