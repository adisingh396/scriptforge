"""Prepare IAM word-level data into ScriptForge canvas format.

Input layout (Kaggle "iam_words"):  <iam>/iam_words/words/a01/a01-000u/a01-000u-00-00.png
Annotations:                        <iam>/words_new.txt   (standard IAM words.txt format)
    id  status  graylevel  ncomp  x y w h  grammar  transcription

Output (data/iam): {split}_imgs.npy (uint8 memmap), {split}_texts.txt, {split}_index.npy, meta.json
Split is WRITER-disjoint (IAM writer = first 3 digits of the form id), so the recognizer/generator
never see a test writer's hand during training.
"""

import argparse
import glob
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

# Characters we keep. Everything else (rare symbols, accents we cannot render) drops the sample.
ALPHABET = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,'-&()!?/ ")


def parse_annotations(path):
    """Return {word_id: transcription} for status==ok entries."""
    out = {}
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            parts = line.split(" ")
            if len(parts) < 9:
                continue
            word_id, status = parts[0], parts[1]
            if status != "ok":
                continue
            text = " ".join(parts[8:])
            out[word_id] = text
    return out


def writer_of(word_id):
    parts = word_id.split("-")
    return parts[0][:3] if len(parts) > 1 else "000"


def build_index(img_root, annotations):
    """id -> png path, keeping only ids that have annotations."""
    paths = {}
    for p in glob.iglob(os.path.join(img_root, "*", "*", "*.png")):
        wid = os.path.splitext(os.path.basename(p))[0]
        if wid in annotations:
            paths[wid] = p
    return paths


def make_canvas(path, height, width, pad=4, normalize=True):
    try:
        im = Image.open(path).convert("L")
    except Exception:
        return None
    arr = np.array(im, dtype=np.uint8)
    mask = arr < 200
    if not mask.any():
        return None
    ys, xs = np.where(mask)
    crop = arr[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    if normalize:
        # IAM scans vary a lot in ink lightness; stretch so paper is 255 and ink is ~0
        lo = float(np.percentile(crop, 2))
        hi = float(np.percentile(crop, 98))
        if hi - lo > 12:
            crop = np.clip((crop.astype(np.float32) - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)
    h, w = crop.shape
    th, tw = height - 2 * pad, width - 2 * pad
    scale = min(th / max(1, h), tw / max(1, w))
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    resized = np.array(Image.fromarray(crop).resize((nw, nh), Image.LANCZOS))
    canvas = np.full((height, width), 255, dtype=np.uint8)
    y0 = (height - nh) // 2
    x0 = (width - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--iam", default=r"C:\Users\adity\Downloads\iam_words",
                    help="folder containing iam_words/words and words_new.txt (or words.txt)")
    ap.add_argument("--img-root", default="", help="override word image root (folder of a01/, a02/ ...)")
    ap.add_argument("--annotations", default="", help="override annotation txt path")
    ap.add_argument("--out", default="data/iam")
    ap.add_argument("--height", type=int, default=40)
    ap.add_argument("--width", type=int, default=192)
    ap.add_argument("--valid-frac", type=float, default=0.10, help="fraction of WRITERS held out for validation")
    ap.add_argument("--test-frac", type=float, default=0.10, help="fraction of WRITERS held out for test")
    ap.add_argument("--max-len", type=int, default=0, help="drop words longer than N chars (0 = auto)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    img_root = args.img_root or os.path.join(args.iam, "iam_words", "words")
    ann_path = args.annotations
    if not ann_path:
        for cand in ("words_new.txt", "words.txt"):
            p = os.path.join(args.iam, cand)
            if os.path.exists(p):
                ann_path = p
                break
    if not os.path.exists(img_root):
        raise SystemExit(f"word image root not found: {img_root}")
    if not ann_path or not os.path.exists(ann_path):
        raise SystemExit("annotations (words_new.txt / words.txt) not found")

    print(f"image root : {img_root}", flush=True)
    print(f"annotations: {ann_path}", flush=True)
    ann = parse_annotations(ann_path)
    print(f"annotated ok words: {len(ann)}", flush=True)

    paths = build_index(img_root, ann)
    print(f"matched png files : {len(paths)}", flush=True)

    ids = sorted(paths)
    if args.limit:
        ids = ids[:args.limit]

    # keep only samples fully representable in our alphabet and not too long
    max_len = args.max_len or max(1, args.width // 4 - 2)
    keep = []
    for wid in ids:
        t = ann[wid].strip()
        if not t or len(t) > max_len:
            continue
        if any(c not in ALPHABET for c in t):
            continue
        keep.append(wid)
    print(f"usable samples    : {len(keep)} (dropped {len(ids) - len(keep)})", flush=True)

    # writer-disjoint split
    writers = sorted({writer_of(w) for w in keep})
    n_val = max(1, int(len(writers) * args.valid_frac))
    n_test = max(1, int(len(writers) * args.test_frac))
    val_w = set(writers[:n_val])
    test_w = set(writers[len(writers) - n_test:])
    splits = {"train": [], "validation": [], "test": []}
    for wid in keep:
        w = writer_of(wid)
        if w in val_w:
            splits["validation"].append(wid)
        elif w in test_w:
            splits["test"].append(wid)
        else:
            splits["train"].append(wid)
    print(f"writers: {len(writers)} | train {len(splits['train'])} "
          f"val {len(splits['validation'])} test {len(splits['test'])}", flush=True)

    os.makedirs(args.out, exist_ok=True)
    for split, wids in splits.items():
        if not wids:
            continue
        canvases = []
        texts = []
        idxs = []
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for i, (wid, canvas) in enumerate(zip(
                    wids, pool.map(lambda w: make_canvas(paths[w], args.height, args.width), wids))):
                if canvas is None:
                    continue
                canvases.append(canvas)
                texts.append(ann[wid].strip())
                idxs.append(i)
        n = len(canvases)
        imgs_path = os.path.join(args.out, f"{split}_imgs.npy")
        imgs = np.lib.format.open_memmap(imgs_path, mode="w+", dtype=np.uint8,
                                        shape=(n, args.height, args.width))
        for i, c in enumerate(canvases):
            imgs[i] = c
        imgs.flush()
        del imgs
        with open(os.path.join(args.out, f"{split}_texts.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(texts) + ("\n" if texts else ""))
        np.save(os.path.join(args.out, f"{split}_index.npy"), np.array(idxs, dtype=np.int32))
        print(f"[{split}] {n} images -> {imgs_path}", flush=True)
        for t in texts[:3]:
            print(f"   sample: {t!r}", flush=True)

    meta = {"source": "IAM word-level", "img_root": img_root, "annotations": ann_path,
            "height": args.height, "width": args.width,
            "counts": {k: len(v) for k, v in splits.items()}}
    with open(os.path.join(args.out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print("meta:", json.dumps(meta["counts"]), flush=True)


if __name__ == "__main__":
    main()
