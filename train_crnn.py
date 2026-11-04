"""Train the CRNN handwriting recognizer with CTC loss."""

import argparse
import os
import sys
import time
import signal

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.amp import autocast, GradScaler

from sf_data import WordDataset
from sf_models import CRNN, build_charset, text_to_ids, greedy_ctc_decode, crnn_log_probs

CHARSET = build_charset()
BLANK = len(CHARSET)


def set_charset(cs):
    """Override the module charset (IAM uses mixed case + punctuation)."""
    global CHARSET, BLANK
    CHARSET = cs
    BLANK = len(CHARSET)


def uppercase_dataset_texts(ds):
    """Align labels with text_to_ids()'s uppercase convention.

    Without this, validation compares upper-case predictions against mixed-case labels
    and CER looks ~1.0 even for a correct model.
    """
    ds.texts = [t.upper() for t in ds.texts]
    return ds


def derive_charset(texts_path):
    """Charset of the *model's* label space.

    text_to_ids() uppercases, so the charset must be upper-case too; otherwise the label
    space carries case-duplicate classes (72 instead of ~40) and CTC bootstraps poorly.
    """
    chars = set()
    with open(texts_path, "r", encoding="utf-8") as f:
        for line in f:
            chars.update(line.rstrip("\n").upper())
    chars.discard("\n")
    return "".join(sorted(chars))
T_DEFAULT = None


def edit_distance(a, b):
    if a == b:
        return 0
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur.append(min(prev[j] + 1, cur[-1] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[-1]


def collate(batch):
    imgs = torch.stack([b["real"] for b in batch])
    texts = [b["text"] for b in batch]
    targets = []
    lengths = []
    for t in texts:
        ids = text_to_ids(t, CHARSET)
        targets.extend(ids)
        lengths.append(len(ids))
    targets_t = torch.tensor(targets, dtype=torch.long) if targets else torch.zeros(0, dtype=torch.long)
    lengths_t = torch.tensor(lengths, dtype=torch.long)
    return imgs, targets_t, lengths_t, texts


def filter_indices(texts, T):
    out = []
    for i, t in enumerate(texts):
        ids = text_to_ids(t, CHARSET)
        if len(ids) == 0:
            continue
        if len(ids) > T - 2:
            continue
        if len(ids) != len(t.replace(" ", " ")) :
            pass
        # ensure all chars in charset
        if any(c not in CHARSET for c in str(t)):
            continue
        out.append(i)
    return out


def make_loaders(args):
    height = args.height
    width = args.width
    T = width // 4
    train_imgs = os.path.join(args.data_dir, "train_imgs.npy")
    train_texts = os.path.join(args.data_dir, "train_texts.txt")

    # validation dir resolution
    if args.val_dir:
        val_imgs = os.path.join(args.val_dir, "validation_imgs.npy")
        val_texts = os.path.join(args.val_dir, "validation_texts.txt")
        if not (os.path.exists(val_imgs) and os.path.exists(val_texts)):
            # maybe train-style names
            val_imgs = os.path.join(args.val_dir, "train_imgs.npy")
            val_texts = os.path.join(args.val_dir, "train_texts.txt")
    else:
        val_imgs = os.path.join(args.data_dir, "validation_imgs.npy")
        val_texts = os.path.join(args.data_dir, "validation_texts.txt")

    has_val = os.path.exists(val_imgs) and os.path.exists(val_texts)

    train_ds_full = uppercase_dataset_texts(
        WordDataset(train_imgs, train_texts, height, width,
                    augment=not getattr(args, "no_augment", False)))
    train_idx = filter_indices(train_ds_full.texts, T)
    train_ds = Subset(train_ds_full, train_idx)

    if has_val:
        val_ds_full = uppercase_dataset_texts(
            WordDataset(val_imgs, val_texts, height, width, augment=False))
        val_idx = filter_indices(val_ds_full.texts, T)
        val_ds = Subset(val_ds_full, val_idx)
    else:
        # carve last 300 of smoke set as val
        base_ds = WordDataset(train_imgs, train_texts, height, width, augment=False)
        idx = list(range(len(base_ds.texts)))
        val_idx_raw = idx[-300:]
        val_idx = [i for i in val_idx_raw if i in set(filter_indices(base_ds.texts, T))]
        val_ds = Subset(base_ds, val_idx)
        # remove val indices from train
        val_set = set(val_idx)
        train_idx = [i for i in train_idx if i not in val_set]
        train_ds = Subset(train_ds_full, train_idx)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                             num_workers=2, pin_memory=True, drop_last=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=2, pin_memory=True, collate_fn=collate)

    # short-word curriculum loader: CTC escapes blank/length-prior states fast on easy words
    short_len = int(getattr(args, "short_max_len", 5) or 5)
    short_idx = [i for i in train_idx if len(train_ds_full.texts[i].split()) == 1
                 and 0 < len(train_ds_full.texts[i]) <= short_len]
    if len(short_idx) >= args.batch_size * 2:
        short_ds = Subset(train_ds_full, short_idx)
        short_loader = DataLoader(short_ds, batch_size=args.batch_size, shuffle=True,
                                  num_workers=2, pin_memory=True, drop_last=True,
                                  collate_fn=collate)
    else:
        short_loader = train_loader
    print(f"curriculum: {len(short_idx)} words <= {short_len} chars for the first "
          f"{getattr(args, 'short_steps', 0)} steps", flush=True)
    return train_loader, val_loader, T, short_loader


def eval_val(model, val_loader, device, T, show=5):
    model.eval()
    total_ed = 0
    total_chars = 0
    word_correct = 0
    n = 0
    shown = []
    with torch.no_grad():
        for imgs, targets, lengths, texts in val_loader:
            imgs = imgs.to(device)
            lp = crnn_log_probs(imgs, model)  # (B,T,C)
            preds = greedy_ctc_decode(lp, CHARSET)
            for p, t in zip(preds, texts):
                total_ed += edit_distance(p, t)
                total_chars += max(1, len(t))
                if p == t:
                    word_correct += 1
                n += 1
                if len(shown) < show:
                    shown.append((p, t))
    cer = total_ed / max(1, total_chars)
    word_acc = word_correct / max(1, n)
    for p, t in shown:
        flag = "OK " if p == t else "   "
        print(f"   {flag}pred={p!r:20s} target={t!r}", flush=True)
    return cer, word_acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/prepared")
    ap.add_argument("--val-dir", default=None)
    ap.add_argument("--no-augment", action="store_true",
                    help="disable training augmentation")
    ap.add_argument("--grad-clip", type=float, default=5.0,
                    help="grad-norm clip; too tight (<=1) can stall CNN feature learning")
    ap.add_argument("--no-scheduler", action="store_true",
                    help="constant LR instead of OneCycle")
    ap.add_argument("--short-steps", type=int, default=1200,
                    help="steps spent on the short-word curriculum (0 = disable); this escapes CTC blank/length-prior states")
    ap.add_argument("--short-max-len", type=int, default=5,
                    help="max characters of a word used during the curriculum phase")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--time-budget-minutes", type=float, default=None)
    ap.add_argument("--height", type=int, default=48)
    ap.add_argument("--width", type=int, default=192)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--warmup-frac", type=float, default=0.3,
                    help="fraction of total steps spent warming up (higher = safer against CTC blank collapse)")
    ap.add_argument("--charset", default="auto",
                    help="'auto' = derive from training texts, or an explicit character set string")
    ap.add_argument("--out", default="checkpoints/crnn_best.pt")
    args = ap.parse_args()

    # allow batch_size alias
    if not hasattr(args, "batch_size"):
        args.batch_size = 32

    # charset must be known before loaders build CTC targets
    if args.charset == "auto":
        cs = derive_charset(os.path.join(args.data_dir, "train_texts.txt"))
        set_charset(cs)
        print(f"charset auto-derived: {len(CHARSET)} classes -> {CHARSET!r}", flush=True)
    elif args.charset != "default":
        set_charset(args.charset)
        print(f"charset set: {len(CHARSET)} classes -> {CHARSET!r}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    height = args.height
    width = args.width
    T = width // 4

    train_loader, val_loader, T, short_loader = make_loaders(args)

    model = CRNN(num_classes=len(CHARSET), height=height, width=width).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps_per_epoch = max(1, len(train_loader))
    total_steps = steps_per_epoch * args.epochs
    if args.max_steps is not None:
        total_steps = min(total_steps, args.max_steps)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=total_steps, pct_start=args.warmup_frac)
    ctc_loss = nn.CTCLoss(blank=BLANK, zero_infinity=True)
    scaler = GradScaler("cuda", enabled=device.type == "cuda")

    # fixed 16-sample batch (non-augmented)
    train_imgs = os.path.join(args.data_dir, "train_imgs.npy")
    train_texts = os.path.join(args.data_dir, "train_texts.txt")
    base_ds = uppercase_dataset_texts(
        WordDataset(train_imgs, train_texts, height, width, augment=False))
    valid_idx = filter_indices(base_ds.texts, T)
    fixed_idx = valid_idx[:16]
    fixed_imgs = torch.stack([base_ds[i]["real"] for i in fixed_idx]).to(device)
    fixed_texts = [base_ds[i]["text"] for i in fixed_idx]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    best_cer = float("inf")
    global_step = 0
    start_time = time.time()

    stop_flag = {"stop": False}

    def handle_sigint(sig, frame):
        stop_flag["stop"] = True
        print("\n[ctrl-c] stopping after current step...", flush=True)

    signal.signal(signal.SIGINT, handle_sigint)

    time_budget_sec = args.time_budget_minutes * 60 if args.time_budget_minutes else None

    def time_exceeded():
        if time_budget_sec is None:
            return False
        return (time.time() - start_time) >= time_budget_sec

    try:
        short_iter = iter(short_loader)

        def next_short():
            nonlocal short_iter
            try:
                return next(short_iter)
            except StopIteration:
                short_iter = iter(short_loader)
                return next(short_iter)

        for epoch in range(args.epochs):
            model.train()
            for imgs, targets, lengths, texts in train_loader:
                if global_step < args.short_steps and args.short_steps > 0:
                    imgs, targets, lengths, texts = next_short()
                if stop_flag["stop"] or time_exceeded():
                    raise SystemExit
                if args.max_steps is not None and global_step >= args.max_steps:
                    raise SystemExit

                imgs = imgs.to(device)
                targets = targets.to(device)
                lengths = lengths.to(device)

                optimizer.zero_grad(set_to_none=True)
                with autocast("cuda", enabled=device.type == "cuda", dtype=torch.float16):
                    log_probs = crnn_log_probs(imgs, model)  # (B,T,C)
                    lp = log_probs.permute(1, 0, 2)  # (T,B,C)
                    B = imgs.shape[0]
                    input_lengths = torch.full((B,), T, dtype=torch.long, device=device)
                    loss = ctc_loss(lp, targets, input_lengths, lengths)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                if not args.no_scheduler:
                    scheduler.step()

                global_step += 1

                if global_step % 100 == 0:
                    model.eval()
                    with torch.no_grad():
                        flp = crnn_log_probs(fixed_imgs, model)
                        fpreds = greedy_ctc_decode(flp, CHARSET)
                    acc = sum(1 for p, t in zip(fpreds, fixed_texts) if p == t) / len(fixed_texts)
                    print(f"step {global_step} loss {loss.item():.4f} fixed_acc {acc:.3f}", flush=True)
                    model.train()

                if args.max_steps is not None and global_step >= args.max_steps:
                    break

            # end epoch: eval val CER
            cer, wacc = eval_val(model, val_loader, device, T)
            print(f"epoch {epoch} val CER {cer:.4f} val word_acc {wacc:.4f}", flush=True)
            if cer < best_cer:
                best_cer = cer
                torch.save({"state_dict": model.state_dict(), "charset": CHARSET,
                            "height": height, "width": width}, args.out)
                print(f"saved best (CER {cer:.4f}) -> {args.out}", flush=True)

            if stop_flag["stop"] or time_exceeded():
                break
            if args.max_steps is not None and global_step >= args.max_steps:
                break

    except SystemExit:
        pass

    # final eval + save
    cer, wacc = eval_val(model, val_loader, device, T)
    print(f"FINAL val CER {cer:.4f} val word_acc {wacc:.4f}", flush=True)
    if cer < best_cer:
        best_cer = cer
        torch.save({"state_dict": model.state_dict(), "charset": CHARSET,
                    "height": height, "width": width}, args.out)
        print(f"saved best (CER {cer:.4f}) -> {args.out}", flush=True)

    print(f"done. steps={global_step} best_CER={best_cer:.4f}", flush=True)


if __name__ == "__main__":
    main()
