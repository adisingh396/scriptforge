"""ScriptForge conditional GAN training.

pix2pix-style cGAN: font-glyph condition + style latent z -> handwritten word image.
Losses: hinge adversarial + L1 (+ optional CTC legibility through a frozen CRNN).
All inference artifacts use the EMA copy of the generator.
"""

import argparse
import csv
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from sf_data import WordDataset
from sf_ema import ModelEma
from sf_models import (
    CHARSET, CRNN, PatchDiscriminator, StyleEncoder, UNetGenerator,
    crnn_log_probs, text_to_ids,
)

FIXED_TEXTS = ["Simon", "the", "Balthazar", "Mary", "quick", "Anna", "brown", "fox"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default="data/prepared")
    p.add_argument("--limit", type=int, default=0, help="truncate dataset to N samples (0=all)")
    p.add_argument("--steps", type=int, default=30000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--height", type=int, default=32)
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--base", type=int, default=32, help="generator/discriminator base channel width")
    p.add_argument("--cond-font", default="cursive",
                   help="'cursive' | 'block' | explicit .ttf path — font used to rasterise the condition")
    p.add_argument("--cond-case", default="keep",
                   choices=["keep", "title", "upper", "lower"],
                   help="case transform applied when rasterising the condition (keep = as in the label)")
    p.add_argument("--lr-g", type=float, default=2e-4)
    p.add_argument("--lr-d", type=float, default=2e-4)
    p.add_argument("--l1-weight", type=float, default=50.0)
    p.add_argument("--adv", choices=["hinge", "bce"], default="bce",
                   help="adversarial loss; hinge can stall at D=0, bce is the pix2pix default")
    p.add_argument("--style-dim", type=int, default=16)
    p.add_argument("--r1-weight", type=float, default=10.0)
    p.add_argument("--r1-every", type=int, default=16)
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--ctc-weight", type=float, default=0.0)
    p.add_argument("--crnn", default="", help="path to CRNN checkpoint for CTC legibility loss")
    p.add_argument("--ctc-after", type=int, default=8000)
    p.add_argument("--sn", type=int, default=0,
                   help="1 = spectral-norm D (can collapse to a constant under fp16 autocast)")
    p.add_argument("--reset-d", type=int, default=0,
                   help="1 = reinitialise D on resume (use when the old D collapsed)")
    p.add_argument("--sample-every", type=int, default=500)
    p.add_argument("--save-every", type=int, default=2000)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--time-budget-minutes", type=float, default=0.0)
    p.add_argument("--resume", default="")
    p.add_argument("--out", default="checkpoints/gan_last.pt")
    p.add_argument("--log", default="logs/train_gan.csv")
    p.add_argument("--sample-dir", default="outputs/samples")
    return p.parse_args()


def hinge_d_loss(d_real, d_fake):
    return F.relu(1.0 - d_real).mean() + F.relu(1.0 + d_fake).mean()


def bce_d_loss(d_real, d_fake, smooth=0.95):
    real_t = torch.full_like(d_real, smooth)
    fake_t = torch.zeros_like(d_fake)
    return F.binary_cross_entropy_with_logits(d_real, real_t) + \
        F.binary_cross_entropy_with_logits(d_fake, fake_t)


def bce_g_adv(d_fake):
    return F.binary_cross_entropy_with_logits(d_fake, torch.ones_like(d_fake))


def r1_penalty(d_out_real, real_imgs):
    """R1 gradient penalty.

    A PatchGAN emits many patch logits, so the scalarisation must average over patches
    (using .sum() inflates the penalty by the number of patches squared and annihilates D).
    """
    grads = torch.autograd.grad(
        outputs=d_out_real.mean(), inputs=real_imgs, create_graph=True
    )[0]
    return grads.pow(2).reshape(grads.shape[0], -1).sum(1).mean()


def save_grid(images, texts, path, cols=4):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(images)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 1.1))
    axes = np.array(axes).reshape(-1)
    for i, ax in enumerate(axes):
        ax.axis("off")
        if i < n:
            img = images[i]
            img = np.clip((img + 1.0) * 127.5, 0, 255).astype(np.uint8)
            ax.imshow(img, cmap="gray", vmin=0, vmax=255)
            ax.set_title(texts[i], fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def make_fixed_inputs(args, device, font_path=None, font_case=None):
    from sf_render import render_text
    glyph = np.stack([
        render_text(t, args.height, args.width, font_path=font_path, case=font_case).astype(np.float32)
        / 127.5 - 1.0
        for t in FIXED_TEXTS
    ])  # (N,H,W)
    g = torch.tensor(glyph, dtype=torch.float32).unsqueeze(1)  # (N,1,H,W)
    g_d = F.max_pool2d(g, 3, stride=1, padding=1)              # dilated glyph
    cond = torch.cat([g, g_d], dim=1).to(device)               # (N,2,H,W)
    gen = torch.Generator().manual_seed(1234)
    z = torch.randn(len(FIXED_TEXTS), args.style_dim, generator=gen).to(device)
    return cond, z


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    for d in (os.path.dirname(args.out) or ".", os.path.dirname(args.log) or ".", args.sample_dir):
        os.makedirs(d, exist_ok=True)

    # ---------- data ----------
    from sf_render import find_font
    font_path = args.cond_font
    if font_path in ("cursive", "block"):
        font_path = find_font(font_path)
    font_case = None if args.cond_case in ("keep", "", None) else args.cond_case
    print(f"condition: font={font_path or 'PIL default'} case={args.cond_case}", flush=True)

    train_imgs = os.path.join(args.data_dir, "train_imgs.npy")
    train_texts = os.path.join(args.data_dir, "train_texts.txt")
    ds = WordDataset(train_imgs, train_texts, args.height, args.width, augment=True,
                     font_path=font_path, font_case=font_case)
    if args.limit and args.limit > 0:
        n = min(args.limit, len(ds))
        ds.imgs = ds.imgs[:n]
        ds.texts = ds.texts[:n]
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=True,
                        drop_last=True, persistent_workers=args.num_workers > 0)
    print(f"data: {len(ds)} samples | device: {device} | canvas {args.height}x{args.width}", flush=True)

    # ---------- models ----------
    G = UNetGenerator(cond_ch=2, style_ch=8, base=args.base).to(device)
    D = PatchDiscriminator(cond_ch=2, base=args.base, use_sn=bool(args.sn)).to(device)
    S = StyleEncoder(z_dim=args.style_dim).to(device)
    ema = ModelEma(G, base_decay=args.ema_decay)
    g_params = list(G.parameters()) + list(S.parameters())
    g_opt = torch.optim.Adam(g_params, lr=args.lr_g, betas=(0.5, 0.999))
    d_opt = torch.optim.Adam(D.parameters(), lr=args.lr_d, betas=(0.5, 0.999))
    scaler_g = torch.amp.GradScaler("cuda")
    scaler_d = torch.amp.GradScaler("cuda")
    print(f"params: G={sum(p.numel() for p in G.parameters())/1e6:.2f}M "
          f"D={sum(p.numel() for p in D.parameters())/1e6:.2f}M", flush=True)

    # ---------- optional CRNN legibility loss ----------
    crnn = None
    ctc_loss_fn = None
    crnn_hw = None
    T = args.width // 4
    if args.ctc_weight > 0 and args.crnn:
        ck = torch.load(args.crnn, map_location=device, weights_only=False)
        crnn = CRNN(len(ck["charset"]), height=ck["height"], width=ck["width"]).to(device)
        crnn.load_state_dict(ck["state_dict"])
        # cuDNN LSTM backward needs train() mode, but BatchNorm must keep its running
        # stats (frozen oracle), so put norm layers back into eval.
        crnn.train()
        for m in crnn.modules():
            if isinstance(m, (torch.nn.BatchNorm2d, torch.nn.InstanceNorm2d)):
                m.eval()
        for p in crnn.parameters():
            p.requires_grad_(False)
        ctc_loss_fn = torch.nn.CTCLoss(blank=len(ck["charset"]), zero_infinity=True)
        crnn_hw = (int(ck["height"]), int(ck["width"]))
        T = crnn_hw[1] // 4  # recognizer's own time axis
        print(f"CRNN legibility loss enabled (weight {args.ctc_weight}, after step {args.ctc_after}) "
              f"| recognizer canvas {crnn_hw[0]}x{crnn_hw[1]}", flush=True)

    # ---------- fixed sample inputs ----------
    fixed_cond, fixed_z = make_fixed_inputs(args, device, font_path=font_path, font_case=font_case)

    # ---------- resume ----------
    start_step = 0
    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        G.load_state_dict(ck["G"]); S.load_state_dict(ck["S"]); ema.load_state_dict(ck["G_ema"])
        g_opt.load_state_dict(ck["G_opt"])
        scaler_g.load_state_dict(ck["scaler_g"])
        if args.reset_d:
            print("reset-d: discriminator reinitialised fresh (old one had collapsed)", flush=True)
            d_opt = torch.optim.Adam(D.parameters(), lr=args.lr_d, betas=(0.5, 0.999))
        else:
            D.load_state_dict(ck["D"]); d_opt.load_state_dict(ck["D_opt"])
            scaler_d.load_state_dict(ck["scaler_d"])
        start_step = ck["step"]
        torch.set_rng_state(ck["rng_cpu"].cpu())
        if device.type == "cuda" and ck.get("rng_cuda") is not None:
            torch.cuda.set_rng_state_all([s.cpu() for s in ck["rng_cuda"]])
        # optimizer state restores its own lr; honour this run's rates instead
        for grp in g_opt.param_groups:
            grp["lr"] = args.lr_g
        for grp in d_opt.param_groups:
            grp["lr"] = args.lr_d
        print(f"resumed from {args.resume} at step {start_step}", flush=True)

    def save_ckpt(step):
        torch.save({
            "G": G.state_dict(), "G_ema": ema.state_dict(), "D": D.state_dict(),
            "S": S.state_dict(), "G_opt": g_opt.state_dict(), "D_opt": d_opt.state_dict(),
            "scaler_g": scaler_g.state_dict(), "scaler_d": scaler_d.state_dict(),
            "step": step, "charset": CHARSET,
            "height": args.height, "width": args.width, "style_dim": args.style_dim,
            "base": args.base,
            "args": vars(args),
            "rng_cpu": torch.get_rng_state(),
            "rng_cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
        }, args.out)
        print(f"saved checkpoint -> {args.out} (step {step})", flush=True)

    # ---------- log ----------
    log_exists = os.path.exists(args.log)
    log_f = open(args.log, "a", newline="", encoding="utf-8")
    log_w = csv.writer(log_f)
    if not log_exists:
        log_w.writerow(["step", "d_loss", "g_loss", "adv", "l1", "ctc", "d_real", "d_fake", "elapsed_min"])

    t0 = time.time()
    step = start_step
    stop = False
    data_iter = iter(loader)

    def next_batch():
        nonlocal data_iter
        try:
            return next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            return next(data_iter)

    # ctrl-c safe
    import signal
    def _sigint(sig, frame):
        nonlocal stop
        stop = True
        print("\n[ctrl-c] stopping after current step...", flush=True)
    signal.signal(signal.SIGINT, _sigint)

    while step < args.steps and not stop:
        batch = next_batch()
        real = batch["real"].to(device, non_blocking=True)
        cond = batch["cond"].to(device, non_blocking=True)
        texts = batch["text"]
        B = real.shape[0]

        z = torch.randn(B, args.style_dim, device=device)
        style_map = S(z, args.height, args.width)

        # ----- D step -----
        d_opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            d_real = D(real, cond)
            fake = G(cond, style_map)
            d_fake = D(fake.detach(), cond)
            if args.adv == "hinge":
                d_loss = hinge_d_loss(d_real, d_fake)
            else:
                d_loss = bce_d_loss(d_real, d_fake)
        scaler_d.scale(d_loss).backward()
        if args.r1_weight > 0 and (step % args.r1_every == 0):
            real.requires_grad_(True)
            d_out = D(real, cond)
            r1 = r1_penalty(d_out, real) * (args.r1_weight * args.r1_every)
            scaler_d.scale(r1).backward()
            real.requires_grad_(False)
        scaler_d.step(d_opt)
        scaler_d.update()

        # ----- G step -----
        g_opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            fake = G(cond, style_map)
            d_fake = D(fake, cond)
            adv = -d_fake.mean() if args.adv == "hinge" else bce_g_adv(d_fake)
            l1 = F.l1_loss(fake, real)
            g_loss = adv + args.l1_weight * l1
            ctc_val = 0.0
            if crnn is not None and step >= args.ctc_after:
                keep = [i for i, t in enumerate(texts) if 0 < len(text_to_ids(t)) <= T - 2]
                if keep:
                    idx = torch.tensor(keep, device=device)
                    inp = fake[idx]
                    if crnn_hw is not None and (inp.shape[2], inp.shape[3]) != crnn_hw:
                        inp = F.interpolate(inp, size=crnn_hw, mode="bilinear", align_corners=False)
                    lp = crnn_log_probs(inp, crnn).permute(1, 0, 2)  # (T,b,C)
                    targets = torch.cat([torch.tensor(text_to_ids(texts[i]), device=device) for i in keep])
                    lens = torch.tensor([len(text_to_ids(texts[i])) for i in keep], device=device)
                    inp_lens = torch.full((len(keep),), T, dtype=torch.long, device=device)
                    ctc = ctc_loss_fn(lp, targets, inp_lens, lens)
                    if torch.isfinite(ctc):
                        g_loss = g_loss + args.ctc_weight * ctc
                        ctc_val = float(ctc)
        scaler_g.scale(g_loss).backward()
        scaler_g.step(g_opt)
        scaler_g.update()
        ema.update(G)

        step += 1

        if step % args.log_every == 0:
            elapsed = (time.time() - t0) / 60.0
            d_r = float(d_real.mean()); d_f = float(d_fake.mean())
            log_w.writerow([step, f"{d_loss.item():.4f}", f"{g_loss.item():.4f}",
                            f"{adv.item():.4f}", f"{l1.item():.4f}", f"{ctc_val:.4f}",
                            f"{d_r:.3f}", f"{d_f:.3f}", f"{elapsed:.1f}"])
            log_f.flush()
            print(f"step {step}/{args.steps} d={d_loss.item():.3f} g={g_loss.item():.3f} "
                  f"l1={l1.item():.3f} ctc={ctc_val:.3f} d_real={d_r:.2f} d_fake={d_f:.2f}", flush=True)

        if step % args.sample_every == 0:
            Gs = ema.model
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
                sm = S(fixed_z, args.height, args.width)
                out = Gs(fixed_cond, sm)
            imgs = out.float().squeeze(1).cpu().numpy()
            save_grid(imgs, FIXED_TEXTS, os.path.join(args.sample_dir, f"step_{step:06d}.png"))
            print(f"saved sample grid -> {args.sample_dir}\\step_{step:06d}.png", flush=True)

        if step % args.save_every == 0:
            save_ckpt(step)

        if args.time_budget_minutes > 0 and (time.time() - t0) / 60.0 > args.time_budget_minutes:
            print("time budget reached", flush=True)
            break

    save_ckpt(step)
    log_f.close()
    print(f"done. steps trained this session: {step - start_step}", flush=True)


if __name__ == "__main__":
    main()
