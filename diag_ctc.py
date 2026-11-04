"""Diagnostic: can the CRNN overfit 200 IAM words? (blank-collapse check)

    python diag_ctc.py --data-dir data/iam --height 40 --width 192 --steps 300 --lr 3e-4
"""
import argparse
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, "src")
from sf_data import WordDataset
from sf_models import CRNN, crnn_log_probs, greedy_ctc_decode, text_to_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/iam")
    ap.add_argument("--height", type=int, default=40)
    ap.add_argument("--width", type=int, default=192)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = WordDataset(os.path.join(args.data_dir, "train_imgs.npy"),
                     os.path.join(args.data_dir, "train_texts.txt"),
                     args.height, args.width, augment=False)
    ds.texts = [t.upper() for t in ds.texts]  # match text_to_ids()'s uppercase convention
    charset = "".join(sorted({c for t in ds.texts for c in t}))
    blank = len(charset)
    T = args.width // 4
    print(f"charset {len(charset)} | T {T} | device {dev}", flush=True)

    idx = [i for i in range(len(ds))
           if 0 < len(text_to_ids(ds.texts[i], charset)) <= T - 2][:args.n]
    imgs = torch.stack([ds[i]["real"] for i in idx]).to(dev)
    texts = [ds.texts[i] for i in idx]
    targets = torch.cat([torch.tensor(text_to_ids(t, charset)) for t in texts]).to(dev)
    lengths = torch.tensor([len(text_to_ids(t, charset)) for t in texts])
    print(f"overfit set: {len(idx)} words | e.g. {texts[:5]}", flush=True)

    model = CRNN(len(charset), height=args.height, width=args.width).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    ctc = nn.CTCLoss(blank=blank, zero_infinity=True)
    scaler = torch.amp.GradScaler("cuda")

    for step in range(1, args.steps + 1):
        model.train()
        sel = torch.randint(0, len(idx), (min(args.batch, len(idx)),))
        bi = imgs[sel]
        bt = [texts[i] for i in sel.tolist()]
        tgt = torch.cat([torch.tensor(text_to_ids(t, charset)) for t in bt]).to(dev)
        tlen = torch.tensor([len(text_to_ids(t, charset)) for t in bt])
        with torch.amp.autocast("cuda", dtype=torch.float16):
            lp = crnn_log_probs(bi, model).permute(1, 0, 2)
            loss = ctc(lp, tgt, torch.full((len(bt),), T, dtype=torch.long), tlen)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(opt)
        scaler.update()

        if step % 25 == 0:
            model.eval()
            with torch.no_grad():
                lp = crnn_log_probs(imgs[:16], model)
            preds = greedy_ctc_decode(lp, charset)
            exact = sum(p == t for p, t in zip(preds, texts[:16]))
            print(f"step {step} loss {loss.item():.3f} | exact {exact}/16 | "
                  f"pred[0]={preds[0]!r} target[0]={texts[0]!r}", flush=True)
            model.train()

    model.eval()
    with torch.no_grad():
        preds = greedy_ctc_decode(crnn_log_probs(imgs, model), charset)
    exact = sum(p == t for p, t in zip(preds, texts))
    print(f"FINAL overfit: {exact}/{len(texts)} exact", flush=True)
    for p, t in list(zip(preds, texts))[:5]:
        print(f"  pred={p!r} target={t!r}", flush=True)


if __name__ == "__main__":
    main()
