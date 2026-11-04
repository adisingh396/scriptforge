"""Prepare the Kaggle written-names dataset into ScriptForge canvas format."""

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from PIL import Image

# split -> (subfolder under archive, csv filename)
SPLIT_INFO = {
    "train": ("train_v2/train", "written_name_train_v2.csv"),
    "validation": ("validation_v2/validation", "written_name_validation_v2.csv"),
    "test": ("test_v2/test", "written_name_test_v2.csv"),
}


def _is_valid_label(v):
    if v is None:
        return False
    s = str(v).strip()
    if s == "" or s.lower() == "nan":
        return False
    if s.upper() == "UNREADABLE":
        return False
    return True


def _process_row(task):
    idx, filename, identity, archive, folder, height, width = task
    path = os.path.join(archive, folder, str(filename))
    if not os.path.exists(path):
        return None
    try:
        im = Image.open(path).convert("L")
    except Exception:
        return None
    arr = np.array(im, dtype=np.uint8)
    mask = arr < 200
    if not mask.any():
        return None
    ys, xs = np.where(mask)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    crop = arr[y0:y1, x0:x1]

    target_h = height - 8
    target_w = width - 16
    h, w = crop.shape
    scale = min(target_h / max(1, h), target_w / max(1, w))
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))

    crop_img = Image.fromarray(crop)
    resized = crop_img.resize((new_w, new_h), Image.LANCZOS)
    canvas = np.full((height, width), 255, dtype=np.uint8)
    y_off = (height - new_h) // 2
    x_off = (width - new_w) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = np.array(resized, dtype=np.uint8)

    label = str(identity).strip().upper()
    return canvas, label, int(idx)


def prepare_split(archive, split, limit, height, width, workers, out):
    if split not in SPLIT_INFO:
        raise ValueError(f"unknown split {split}")
    folder, csv_name = SPLIT_INFO[split]
    csv_path = os.path.join(archive, csv_name)
    df = pd.read_csv(csv_path)
    print(f"[{split}] csv rows: {len(df)}", flush=True)

    # filter invalid labels
    valid_mask = df["IDENTITY"].apply(_is_valid_label)
    df = df[valid_mask].copy()
    print(f"[{split}] rows with valid labels: {len(df)}", flush=True)

    # keep original row index for traceability
    df = df.reset_index(drop=False)
    if "index" in df.columns:
        df = df.rename(columns={"index": "ORIG_IDX"})

    if limit and limit > 0:
        df = df.head(limit)

    print(f"[{split}] rows to process (after limit): {len(df)}", flush=True)

    tasks = []
    for row in df.itertuples(index=False):
        tasks.append((int(row.ORIG_IDX), str(row.FILENAME), str(row.IDENTITY),
                      archive, folder, height, width))

    valid_canvases = []
    valid_labels = []
    valid_indices = []

    try:
        from tqdm import tqdm
        iterator = tqdm(range(0, len(tasks), 1000), desc=split)
    except Exception:
        iterator = range(0, len(tasks), 1000)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for start in iterator:
            end = min(start + 1000, len(tasks))
            batch = tasks[start:end]
            for res in pool.map(_process_row, batch):
                if res is None:
                    continue
                canvas, label, orig_idx = res
                valid_canvases.append(canvas)
                valid_labels.append(label)
                valid_indices.append(orig_idx)

    n = len(valid_canvases)
    os.makedirs(out, exist_ok=True)
    imgs_path = os.path.join(out, f"{split}_imgs.npy")
    texts_path = os.path.join(out, f"{split}_texts.txt")
    index_path = os.path.join(out, f"{split}_index.npy")

    if n == 0:
        # empty memmap
        arr = np.zeros((0, height, width), dtype=np.uint8)
        np.save(imgs_path, arr)
        with open(texts_path, "w", encoding="utf-8") as f:
            pass
        np.save(index_path, np.zeros((0,), dtype=np.int32))
    else:
        # open_memmap writes a proper .npy header so np.load(mmap_mode="r") works later
        imgs = np.lib.format.open_memmap(
            imgs_path, mode="w+", dtype=np.uint8, shape=(n, height, width))
        for i, c in enumerate(valid_canvases):
            imgs[i] = c
        imgs.flush()
        del imgs
        with open(texts_path, "w", encoding="utf-8") as f:
            for lab in valid_labels:
                f.write(lab + "\n")
        np.save(index_path, np.array(valid_indices, dtype=np.int32))

    meta = {
        "height": height,
        "width": width,
        "count": n,
        "archive": archive,
        "split": split,
    }
    with open(os.path.join(out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[{split}] prepared {n} images -> {out}")
    # sample pairs
    for i in range(min(5, n)):
        print(f"  sample {i}: {valid_indices[i]} {valid_labels[i]}")
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", default=r"C:\Users\adity\Downloads\archive")
    ap.add_argument("--splits", default="train,validation")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--height", type=int, default=48)
    ap.add_argument("--width", type=int, default=192)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="data/prepared")
    args = ap.parse_args()

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    for sp in splits:
        prepare_split(args.archive, sp, args.limit, args.height, args.width, args.workers, args.out)


if __name__ == "__main__":
    main()
