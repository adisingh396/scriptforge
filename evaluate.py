"""Evaluate ScriptForge: recognizer accuracy on real data + legibility of GAN output.

Usage:
    python evaluate.py                    # uses checkpoints/crnn_best.pt + gan_last.pt
    python evaluate.py --gan-samples 200  # number of generated words to score
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from sf_data import WordDataset
from sf_models import (
    CHARSET, CRNN, StyleEncoder, UNetGenerator, crnn_log_probs, greedy_ctc_decode, text_to_ids,
)
from infer import generate_word


def levenshtein(a, b):
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer_wer(pairs):
    """pairs: list of (pred, target). Returns (CER, word accuracy)."""
    if not pairs:
        return 1.0, 0.0
    err = 0
    tot = 0
    exact = 0
    for pred, tgt in pairs:
        err += levenshtein(pred, tgt)
        tot += max(1, len(tgt))
        if pred == tgt:
            exact += 1
    return err / tot, exact / len(pairs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crnn", default="checkpoints/crnn_best.pt")
    ap.add_argument("--gan", default="checkpoints/gan_last.pt")
    ap.add_argument("--data-dir", default="data/prepared32")
    ap.add_argument("--real-samples", type=int, default=2000, help="real validation images to score")
    ap.add_argument("--gan-samples", type=int, default=200, help="generated words to score")
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- recognizer on REAL validation data ----
    ck = torch.load(args.crnn, map_location=device, weights_only=False)
    crnn = CRNN(len(ck["charset"]), height=ck["height"], width=ck["width"]).to(device)
    crnn.load_state_dict(ck["state_dict"])
    crnn.eval()
    h, w = ck["height"], ck["width"]

    val_ds = WordDataset(os.path.join(args.data_dir, "validation_imgs.npy"),
                         os.path.join(args.data_dir, "validation_texts.txt"),
                         None, None, augment=False)
    n = min(args.real_samples, len(val_ds))
    idx = np.random.default_rng(0).choice(len(val_ds), n, replace=False)
    pairs = []
    with torch.no_grad():
        for s in range(0, n, args.batch):
            ids = idx[s:s + args.batch]
            imgs = torch.stack([val_ds[int(i)]["real"] for i in ids]).to(device)
            if (imgs.shape[2], imgs.shape[3]) != (h, w):  # recognizer has its own native canvas
                imgs = F.interpolate(imgs, size=(h, w), mode="bilinear", align_corners=False)
            texts = [val_ds.texts[int(i)].upper() for i in ids]  # recognizer labels are uppercase
            preds = greedy_ctc_decode(crnn_log_probs(imgs, crnn), ck["charset"])
            pairs += list(zip(preds, texts))
    rcer, rwacc = cer_wer(pairs)
    print(f"[real]     CRNN on {n} validation images: CER {rcer:.4f}  word_acc {rwacc:.4f}", flush=True)

    # ---- GAN legibility ----
    if not os.path.exists(args.gan):
        print(f"[gan] {args.gan} not found, skipping", flush=True)
        return
    gck = torch.load(args.gan, map_location=device, weights_only=False)
    G = UNetGenerator(cond_ch=2, style_ch=8, base=int(gck.get("base", 48)))
    gsd = {k: v for k, v in gck["G_ema"].items() if not k.startswith("_")}
    G.load_state_dict(gsd)
    G.to(device).eval()
    S = StyleEncoder(z_dim=gck.get("style_dim", 16))
    if gck.get("S"):
        S.load_state_dict(gck["S"])
    S.to(device).eval()
    gh, gw = int(gck["height"]), int(gck["width"])

    rng = np.random.default_rng(1)
    sample_idx = rng.choice(len(val_ds), min(args.gan_samples, len(val_ds)), replace=False)
    pairs = []
    with torch.no_grad():
        for i in sample_idx:
            text = val_ds.texts[int(i)]
            target = text.upper()  # compare in the recognizer's label space
            if not (0 < len(text_to_ids(text)) <= gw // 4 - 2):
                continue
            seed = int(rng.integers(0, 10 ** 6))
            gen = torch.Generator().manual_seed(seed)
            z = torch.randn(1, S.fc.in_features, generator=gen).to(device)
            img = generate_word(G, S, text, z, gh, gw, device)
            t = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device) / 127.5 - 1
            if (gh, gw) != (h, w):  # recognizer has its own native canvas
                t = F.interpolate(t, size=(h, w), mode="bilinear", align_corners=False)
            pred = greedy_ctc_decode(crnn_log_probs(t, crnn), ck["charset"])[0]
            pairs.append((pred, target))
    gcer, gwacc = cer_wer(pairs)
    print(f"[generated] CRNN on {len(pairs)} GAN words: CER {gcer:.4f}  word_acc {gwacc:.4f}", flush=True)
    print("\nsummary (markdown):")
    print(f"| metric | value |")
    print(f"|---|---|")
    print(f"| Recognizer CER on real validation | {rcer:.4f} |")
    print(f"| Recognizer word accuracy on real validation | {rwacc:.4f} |")
    print(f"| Generated-word legibility CER | {gcer:.4f} |")
    print(f"| Generated-word legibility accuracy | {gwacc:.4f} |")


if __name__ == "__main__":
    main()
