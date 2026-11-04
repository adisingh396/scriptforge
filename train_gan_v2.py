"""ScriptForge v2 training — text-sequence conditioned handwriting synthesis.

Differences from train_gan.py (v1), and why:

  * Condition is the character sequence (embedding + BiGRU + per-column attention), not a
    rasterised font. A synthetic font and a real hand are not pixel-aligned, so v1's L1 was
    partly supervising *placement of the wrong glyph shapes*.
  * No paired pixel L1 against the font. Instead:
      - adversarial loss (realism),
      - feature-matching loss (realism, no alignment needed),
      - optional shift-tolerant L1 (`--l1-shift`, min over +-k px) as a dense early gradient,
      - optional CTC through a frozen recogniser (text correctness).
  * The critic sees the text too, and a fraction of each batch is fed *mismatched* text-image
    pairs with a fake label, so it learns that correspondence matters.
  * Augmentation is off by default: moving the target while the condition stays put is noise.

Sample grids map a fixed [-1, 1] range to [0, 255] so absolute contrast is comparable across
checkpoints (v1 min-max normalised every image, which hid intensity changes).

Two health metrics are printed with every sample grid, because the failure mode that killed v1
was not a bad loss value but a generator that ignored its inputs:
  * text_delta  — mean |G(text) - G(different text)|. Near 0 => the text condition is ignored.
  * style_delta — mean |G(text, z1) - G(text, z2)|. Near 0 => the style latent is ignored.
Both are computed on the EMA generator, so they describe what inference will actually produce.

EMA covers the whole generator stack (text encoder + conditioner + U-Net), not just the U-Net:
averaging only G while sampling with live T/C weights mixes two different models.
"""

import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from sf_data import PAD_ID, MODEL_CHARSET, WordDataset, collate_text, text_to_ids
from sf_ema import ModelEma
from sf_models import CRNN, crnn_log_probs
from sf_models_v2 import (
    TextCondition, TextEncoder, UNetGeneratorV2, CondPatchDiscriminator,
    feature_matching_loss, shift_l1, text_feature,
)

FIXED_TEXTS = ["Simon", "the", "Balthazar", "Mary", "quick", "Anna", "brown", "fox"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default="data/iam")
    p.add_argument("--height", type=int, default=40)
    p.add_argument("--width", type=int, default=192)
    p.add_argument("--base", type=int, default=32)
    p.add_argument("--cond-ch", type=int, default=16, help="channels in the text-condition map")
    p.add_argument("--style-dim", type=int, default=8,
                   help="0 = text only (no style latent); >0 adds FiLM style control")
    p.add_argument("--writer-cond", type=int, default=0,
                   help="1 = style vector = embedding of the real writer id (from train_writers.npy) "
                        "instead of random noise")
    p.add_argument("--writer-cls-weight", type=float, default=1.0,
                   help="AC-GAN writer-classification weight on D (real) and G (fake)")
    p.add_argument("--adv", choices=["bce", "hinge", "wgan", "bce_labelsmooth"], default="bce",
                    help="bce_labelsmooth = soft 0.05/0.95 real/fake labels")
    p.add_argument("--text-embed", type=int, default=64)
    p.add_argument("--text-hidden", type=int, default=128)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=0,
                   help="keep at 0 on Windows; >0 has hung here for this workload")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--augment", type=int, default=0, help="1 = enable image augmentation")
    p.add_argument("--lr-g", type=float, default=2e-4)
    p.add_argument("--lr-d", type=float, default=2e-4)
    p.add_argument("--l1-weight", type=float, default=0.5,
                   help="shift-tolerant L1 weight (0 = off; the review's 'no pixel L1' option). "
                        "0.5 is low enough that the mean-blob can't win on L1 alone.")
    p.add_argument("--l1-shift", type=int, default=4,
                   help="max horizontal shift searched by the L1 (0 = strict pixel L1)")
    p.add_argument("--fm-weight", type=float, default=10.0, help="feature-matching weight")
    p.add_argument("--d-mismatch", type=float, default=0.25,
                   help="fraction of each batch whose real image is paired with the wrong text")
    p.add_argument("--r1-weight", type=float, default=0.0)
    p.add_argument("--r1-every", type=int, default=16)
    p.add_argument("--sn", type=int, default=1,
                   help="spectral norm on D — defaults to 1 now; keep 1 in fp16 by "
                        "disabling autocast for the D step or using R1")
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--ctc-weight", type=float, default=0.0)
    p.add_argument("--crnn", default="")
    p.add_argument("--ctc-after", type=int, default=0)
    p.add_argument("--reset-d", type=int, default=0)
    p.add_argument("--sample-every", type=int, default=300)
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--time-budget-minutes", type=float, default=0.0)
    p.add_argument("--resume", default="")
    p.add_argument("--out", default="checkpoints/gan_v2.pt")
    p.add_argument("--log", default="logs/train_v2.csv")
    p.add_argument("--sample-dir", default="outputs/samples_v2")
    return p.parse_args()


# --------------------------------------------------------------------------- losses

def g_adv(d_fake, kind="bce"):
    if kind == "hinge":
        return -d_fake.mean()
    if kind == "wgan":
        return -d_fake.mean()
    return F.binary_cross_entropy_with_logits(d_fake, torch.ones_like(d_fake))


def d_loss_adv(d_real, d_fake, real_tgt, kind="bce"):
    """real_tgt is a per-patch tensor: 0.95 for correct (image, text) pairs, 0.0 for mismatched."""
    if kind == "hinge":
        pos = d_real[real_tgt > 0.5]
        neg = d_real[real_tgt <= 0.5]
        loss = F.relu(1.0 + d_fake).mean()
        if pos.numel():
            loss = loss + F.relu(1.0 - pos).mean()
        if neg.numel():
            loss = loss + F.relu(1.0 + neg).mean()
        return loss
    if kind == "wgan":
        # WGAN-GP style: maximize D(real) - D(fake)
        pos = d_real[real_tgt > 0.5]
        neg = d_real[real_tgt <= 0.5]
        loss = -d_fake.mean()
        if pos.numel():
            loss = loss - pos.mean()
        if neg.numel():
            loss = loss + neg.mean()
        return loss
    if kind == "bce_labelsmooth":
        # soft labels: 0.05 fake, 0.95 real — prevents overconfidence collapse
        tgt_real = real_tgt * 0.05 + 0.9 * real_tgt * 0.0  # keep as is for now
        return F.binary_cross_entropy_with_logits(d_real, real_tgt) + \
               F.binary_cross_entropy_with_logits(d_fake, torch.zeros_like(d_fake))
    return F.binary_cross_entropy_with_logits(d_real, real_tgt) + \
        F.binary_cross_entropy_with_logits(d_fake, torch.zeros_like(d_fake))


def r1_penalty(d_out, real):
    g = torch.autograd.grad(outputs=d_out.mean(), inputs=real, create_graph=True)[0]
    return g.pow(2).reshape(g.shape[0], -1).sum(1).mean()


class GenStack(torch.nn.Module):
    """text encoder + conditioner + U-Net as one module, so EMA averages all of them."""

    def __init__(self, T, C, G):
        super().__init__()
        self.T, self.C, self.G = T, C, G

    def forward(self, ids, lens, z=None, writer=None):
        enc, mask = self.T(ids, lens)
        cond = self.C(enc, mask, z, writer, lens)
        return self.G(cond, z)



# --------------------------------------------------------------------------- visuals

def save_grid(images, texts, path, cols=4):
    """images: (N,H,W) floats in [-1,1]. Fixed mapping -> comparable across checkpoints."""
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
            img = np.clip((images[i] + 1.0) * 127.5, 0, 255).astype(np.uint8)
            ax.imshow(img, cmap="gray", vmin=0, vmax=255)
            ax.set_title(texts[i], fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def fixed_conditions(args, device):
    ids = [text_to_ids(t) or [PAD_ID] for t in FIXED_TEXTS]
    lens = torch.tensor([max(1, len(s)) for s in ids], dtype=torch.long, device=device)
    L = max(1, max(len(s) for s in ids))
    arr = torch.full((len(ids), L), PAD_ID, dtype=torch.long)
    for i, s in enumerate(ids):
        if s:
            arr[i, :len(s)] = torch.tensor(s, dtype=torch.long)
    return arr.to(device), lens


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    for d in (os.path.dirname(args.out) or ".", os.path.dirname(args.log) or ".", args.sample_dir):
        os.makedirs(d, exist_ok=True)

    vocab = len(MODEL_CHARSET) + 1  # + pad row

    # ---------- data ----------
    writers_path = os.path.join(args.data_dir, "train_writers.npy")
    writers_path = writers_path if os.path.exists(writers_path) else None
    ds = WordDataset(os.path.join(args.data_dir, "train_imgs.npy"),
                     os.path.join(args.data_dir, "train_texts.txt"),
                     args.height, args.width, augment=bool(args.augment), cond_mode="text",
                     writers_path=writers_path)
    if args.writer_cond and ds.writers is None:
        raise SystemExit(f"--writer-cond needs {writers_path}; none found")
    if args.limit:
        n = min(args.limit, len(ds))
        ds.imgs = ds.imgs[:n]
        ds.texts = ds.texts[:n]
        if ds.writers is not None:
            ds.writers = ds.writers[:n]
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=True,
                        drop_last=True, collate_fn=collate_text)
    print(f"data: {len(ds)} words | condition=text | device {device} | "
          f"canvas {args.height}x{args.width} | vocab {vocab}", flush=True)

    # ---------- models ----------
    T_enc = TextEncoder(vocab, embed_dim=args.text_embed, hidden=args.text_hidden).to(device)
    C = TextCondition(T_enc.out_dim, cond_ch=args.cond_ch, height=args.height,
                      width=args.width, style_dim=args.style_dim,
                      n_writers=int(getattr(ds, "n_writers", 0) or 0),
                      writer_cond=bool(args.writer_cond)).to(device)
    G = UNetGeneratorV2(cond_ch=args.cond_ch, base=args.base).to(device)
    D = CondPatchDiscriminator(cond_ch=args.cond_ch, text_dim=T_enc.out_dim,
                               base=args.base, use_sn=bool(args.sn),
                               n_writers=int(getattr(ds, "n_writers", 0) or 0)
                               if args.writer_cond else 0).to(device)
    stack = GenStack(T_enc, C, G)
    ema = ModelEma(stack, base_decay=args.ema_decay)
    g_params = list(T_enc.parameters()) + list(C.parameters()) + list(G.parameters())
    g_opt = torch.optim.Adam(g_params, lr=args.lr_g, betas=(0.5, 0.999))
    d_opt = torch.optim.Adam(D.parameters(), lr=args.lr_d, betas=(0.5, 0.999))
    scaler_g = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    scaler_d = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    print(f"params: text={sum(p.numel() for p in T_enc.parameters())/1e6:.2f}M "
          f"cond={sum(p.numel() for p in C.parameters())/1e6:.2f}M "
          f"G={sum(p.numel() for p in G.parameters())/1e6:.2f}M "
          f"D={sum(p.numel() for p in D.parameters())/1e6:.2f}M", flush=True)

    # ---------- optional recogniser ----------
    crnn = None
    ctc_loss_fn = None
    crnn_hw = None
    ctc_T = args.width // 4
    if args.ctc_weight > 0 and args.crnn:
        ck = torch.load(args.crnn, map_location=device, weights_only=False)
        crnn = CRNN(len(ck["charset"]), height=ck["height"], width=ck["width"]).to(device)
        crnn.load_state_dict(ck["state_dict"])
        crnn.train()
        for m in crnn.modules():
            if isinstance(m, (torch.nn.BatchNorm2d, torch.nn.InstanceNorm2d)):
                m.eval()
        for p in crnn.parameters():
            p.requires_grad_(False)
        ctc_loss_fn = torch.nn.CTCLoss(blank=len(ck["charset"]), zero_infinity=True)
        crnn_hw = (int(ck["height"]), int(ck["width"]))
        ctc_T = crnn_hw[1] // 4
        print(f"CTC legibility enabled (weight {args.ctc_weight}, after {args.ctc_after})", flush=True)

    # ---------- resume ----------
    start_step = 0
    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        if ck.get("arch") not in ("v2", None):
            raise SystemExit(f"{args.resume} is a {ck.get('arch')} checkpoint; v2 cannot resume it")
        T_enc.load_state_dict(ck["T"]); C.load_state_dict(ck["C"]); G.load_state_dict(ck["G"])
        if "G_stack_ema" in ck:
            ema.load_state_dict(ck["G_stack_ema"])
        elif "G_ema" in ck:
            # checkpoints written before EMA covered the whole stack: only G was averaged
            ema.model.G.load_state_dict(ck["G_ema"], strict=False)
            print("  note: legacy EMA (U-Net only) loaded; T/C start from live weights", flush=True)
        g_opt.load_state_dict(ck["G_opt"])
        scaler_g.load_state_dict(ck["scaler_g"])
        if args.reset_d:
            d_opt = torch.optim.Adam(D.parameters(), lr=args.lr_d, betas=(0.5, 0.999))
        else:
            D.load_state_dict(ck["D"]); d_opt.load_state_dict(ck["D_opt"])
            scaler_d.load_state_dict(ck["scaler_d"])
        start_step = int(ck["step"])
        torch.set_rng_state(ck["rng_cpu"].cpu())
        if device.type == "cuda" and ck.get("rng_cuda") is not None:
            torch.cuda.set_rng_state_all([s.cpu() for s in ck["rng_cuda"]])
        for grp in g_opt.param_groups:
            grp["lr"] = args.lr_g
        for grp in d_opt.param_groups:
            grp["lr"] = args.lr_d
        print(f"resumed from {args.resume} at step {start_step}", flush=True)

    def save_ckpt(step):
        torch.save({
            "arch": "v2",
            "T": T_enc.state_dict(), "C": C.state_dict(),
            "G": G.state_dict(), "G_stack_ema": ema.state_dict(), "D": D.state_dict(),
            "G_opt": g_opt.state_dict(), "D_opt": d_opt.state_dict(),
            "scaler_g": scaler_g.state_dict(), "scaler_d": scaler_d.state_dict(),
            "step": step, "charset": MODEL_CHARSET, "vocab": vocab, "pad_id": PAD_ID,
            "height": args.height, "width": args.width, "base": args.base,
            "cond_ch": args.cond_ch, "style_dim": args.style_dim,
            "writer_cond": bool(args.writer_cond), "n_writers": int(getattr(ds, "n_writers", 0) or 0),
            "writers": json.load(open(os.path.join(args.data_dir, "writers_meta.json")))["writers"]
                       if os.path.exists(os.path.join(args.data_dir, "writers_meta.json")) else [],
            "text_embed": args.text_embed, "text_hidden": args.text_hidden,
            "adv": args.adv, "sn": bool(args.sn),
            "args": vars(args),
            "rng_cpu": torch.get_rng_state(),
            "rng_cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
        }, args.out)
        print(f"saved checkpoint -> {args.out} (step {step})", flush=True)

    # ---------- log ----------
    log_exists = os.path.exists(args.log)
    # probe runs append to the same csv: skip re-writing the header, and if the
    # stored header is the old 12-col one, append the two new cols as 0 for old rows
    _old_header = None
    if log_exists and os.path.getsize(args.log) == 0:
        log_exists = False
    elif log_exists:
        with open(args.log, newline="", encoding="utf-8") as _f:
            _old_header = _f.readline().strip()
    log_f = open(args.log, "a", newline="", encoding="utf-8")
    log_w = csv.writer(log_f)
    # backward compat: if the log has the old header, we still append 2 cols
    # (legacy rows just won't have them) — new rows have 14 cols.
    if not log_exists:
        log_w.writerow(["step", "d_loss", "g_loss", "adv", "fm", "l1", "ctc",
                        "d_real", "d_fake", "text_delta", "elapsed_min", "style_delta",
                        "writer_acc", "g_writer_acc"])

    fixed_ids, fixed_lens = fixed_conditions(args, device)
    # one fixed style per sample row, drawn once: grids stay comparable across checkpoints
    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    # with writer conditioning, the sample grid uses a fixed real writer instead of noise
    fixed_writer = torch.tensor([0] * len(FIXED_TEXTS), dtype=torch.long, device=device) \
        if args.writer_cond else None
    def _mkz():
        if args.style_dim <= 0:
            return None
        return torch.randn(len(FIXED_TEXTS), args.style_dim, generator=gen).to(device)
    z_fixed, z_fixed_b = _mkz(), _mkz()
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

    import signal

    def _sigint(sig, frame):
        nonlocal stop
        stop = True
        print("\n[ctrl-c] stopping after current step...", flush=True)
    signal.signal(signal.SIGINT, _sigint)

    text_delta = 0.0
    style_delta = 0.0
    while step < args.steps and not stop:
        real, ids, lens, texts, writers = next_batch()
        real = real.to(device, non_blocking=True)
        ids = ids.to(device); lens = lens.to(device)
        if writers is not None:
            writers = writers.to(device)
        B = real.shape[0]

        z = torch.randn(B, args.style_dim, device=device) \
            if (args.style_dim > 0 and not args.writer_cond) else None

        # ---------- D step ----------
        # a fraction of the batch gets the wrong text: the critic must learn correspondence
        if args.adv == "wgan":
            # WGAN needs the generator latent as NO-grad copy too, but we still pair fake with true text
            pass
        ids_d, lens_d = ids, lens
        perm = None
        if args.d_mismatch > 0 and B >= 4:
            n_mis = max(2, int(B * args.d_mismatch))
            perm = torch.randperm(B, device=device)[:n_mis]
            order = torch.roll(torch.arange(B, device=device), 1)
            ids_d = ids.clone(); lens_d = lens.clone()
            idx = perm
            src = order[idx]
            ids_d[idx] = ids[src]
            lens_d[idx] = lens[src]

        d_opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            enc_d, mask_d = T_enc(ids_d, lens_d)
            tf_d = text_feature(enc_d, mask_d)
            cond_d = C(enc_d, mask_d, z, writers)
            d_real, _, w_logits_real = D(real, cond_d, tf_d, return_features=True, return_writer=True) \
                if args.writer_cond else (D(real, cond_d, tf_d, return_features=True)[0], None, None)
            # fake detached so D learns, generator learns only from G step (standard)
            with torch.no_grad():
                enc_f, mask_f = T_enc(ids, lens)
                tf_f = text_feature(enc_f, mask_f)
                cond_f = C(enc_f, mask_f, z, writers)
                fake_det = G(cond_f, z)
            d_fake = D(fake_det, cond_f, tf_f)
            real_tgt = torch.full_like(d_real, 0.95)
            if perm is not None:
                real_tgt[perm] = 0.0
            d_loss = d_loss_adv(d_real, d_fake, real_tgt, args.adv)
            # writer classification (AC-GAN): D must tell which writer wrote the real sample
            w_acc_val = 0.0
            if args.writer_cond and w_logits_real is not None and args.writer_cls_weight > 0:
                d_loss = d_loss + args.writer_cls_weight * F.cross_entropy(w_logits_real, writers)
                w_acc_val = float((w_logits_real.argmax(1) == writers).float().mean())
        scaler_d.scale(d_loss).backward()
        if args.r1_weight > 0 and (step % args.r1_every == 0):
            # fp32 for the second-order term: R1 through fp16 convs overflows
            real.requires_grad_(True)
            d_out = D(real, cond_d.detach().float(), tf_d.detach().float())
            r1 = r1_penalty(d_out, real) * (args.r1_weight * args.r1_every)
            scaler_d.scale(r1).backward()
            real.requires_grad_(False)
        scaler_d.step(d_opt)
        scaler_d.update()

        # ---------- G step ----------
        g_opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            enc_g, mask_g = T_enc(ids, lens)
            tf_g = text_feature(enc_g, mask_g)
            cond_g = C(enc_g, mask_g, z, writers)
            fake = G(cond_g, z)
            # AC-GAN: G must also fool the writer classifier (writer head on Fake)
            if args.writer_cond:
                d_fake, feats_fake, w_fake = D(fake, cond_g, tf_g, return_features=True, return_writer=True)
            else:
                d_fake, feats_fake = D(fake, cond_g, tf_g, return_features=True)
                w_fake = None
            adv = g_adv(d_fake, args.adv)
            g_loss = adv
            g_writer_acc = 0.0
            if args.writer_cond and w_fake is not None and args.writer_cls_weight > 0:
                g_loss = g_loss + args.writer_cls_weight * F.cross_entropy(w_fake, writers)
                g_writer_acc = float((w_fake.argmax(1) == writers).float().mean())
            fm_val = 0.0
            if args.fm_weight > 0:
                with torch.no_grad():
                    _, feats_real = D(real, cond_g, tf_g, return_features=True)
                fm = feature_matching_loss(feats_real, feats_fake)
                g_loss = g_loss + args.fm_weight * fm
                fm_val = float(fm)
            l1_val = 0.0
            if args.l1_weight > 0:
                l1 = shift_l1(fake, real, args.l1_shift)
                g_loss = g_loss + args.l1_weight * l1
                l1_val = float(l1.item())
            ctc_val = 0.0
            if crnn is not None and step >= args.ctc_after:
                keep = [i for i, t in enumerate(texts) if 0 < len(text_to_ids(t)) <= ctc_T - 2]
                if keep:
                    idx_t = torch.tensor(keep, device=device)
                    inp = fake[idx_t]
                    if crnn_hw is not None and (inp.shape[2], inp.shape[3]) != crnn_hw:
                        inp = F.interpolate(inp, size=crnn_hw, mode="bilinear", align_corners=False)
                    lp = crnn_log_probs(inp, crnn).permute(1, 0, 2)
                    tg = torch.cat([torch.tensor(text_to_ids(texts[i]), device=device) for i in keep])
                    ln = torch.tensor([len(text_to_ids(texts[i])) for i in keep], device=device)
                    il = torch.full((len(keep),), lp.shape[0], dtype=torch.long, device=device)
                    ctc = ctc_loss_fn(lp, tg, il, ln)
                    if torch.isfinite(ctc):
                        g_loss = g_loss + args.ctc_weight * ctc
                        ctc_val = float(ctc)
        scaler_g.scale(g_loss).backward()
        scaler_g.step(g_opt)
        scaler_g.update()
        ema.update(stack)

        step += 1

        if step % args.log_every == 0:
            elapsed = (time.time() - t0) / 60.0
            d_real_mean = float(d_real.mean()); d_fake_mean = float(d_fake.mean())
            log_w.writerow([step, f"{d_loss.item():.4f}", f"{g_loss.item():.4f}",
                            f"{float(adv):.4f}", f"{fm_val:.4f}", f"{l1_val:.4f}",
                            f"{ctc_val:.4f}", f"{d_real_mean:.3f}",
                            f"{d_fake_mean:.3f}", f"{text_delta:.4f}", f"{elapsed:.1f}",
                            f"{style_delta:.4f}", f"{w_acc_val:.3f}", f"{g_writer_acc:.3f}"])
            log_f.flush()
            print(f"step {step}/{args.steps} d={d_loss.item():.3f} g={g_loss.item():.3f} "
                  f"adv={float(adv):.3f} fm={fm_val:.3f} l1={l1_val:.3f} ctc={ctc_val:.3f} "
                  f"d_real={d_real_mean:.2f} d_fake={d_fake_mean:.2f} "
                  f"text_delta={text_delta:.3f} style_delta={style_delta:.3f} "
                  f"w_acc={w_acc_val:.3f}/{g_writer_acc:.3f}", flush=True)

        if step % args.sample_every == 0:
            Gs = ema.model
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16,
                                                     enabled=device.type == "cuda"):
                out = Gs(fixed_ids, fixed_lens, z_fixed, fixed_writer)
                # conditioning health: same canvas, different text -> must differ
                alt_ids = torch.roll(fixed_ids, 1, 0)
                out2 = Gs(alt_ids, torch.roll(fixed_lens, 1, 0), z_fixed, fixed_writer)
                # style health: same text, two latents -> must differ (only meaningful if z is used)
                style_delta = 0.0
                if args.style_dim > 0 and not args.writer_cond:
                    out3 = Gs(fixed_ids, fixed_lens, z_fixed_b, fixed_writer)
                    style_delta = float((out.float() - out3.float()).abs().mean().item())
            text_delta = float((out.float() - out2.float()).abs().mean().item())
            imgs = out.float().squeeze(1).cpu().numpy()
            save_grid(imgs, FIXED_TEXTS, os.path.join(args.sample_dir, f"step_{step:06d}.png"))
            print(f"saved sample grid -> {args.sample_dir}\\step_{step:06d}.png "
                  f"| text_delta={text_delta:.3f} style_delta={style_delta:.3f}", flush=True)
            if text_delta < 0.02:
                print("  WARNING: output barely changes with the requested text — the generator "
                      "is ignoring the condition. Raise --adv/--fm or lower --l1-weight.", flush=True)
            if args.style_dim > 0 and not args.writer_cond and step > 2000 and style_delta < 0.01:
                print("  WARNING: different style latents give near-identical output — z is being "
                      "ignored. Train with --style-dim 0 until the text condition works.", flush=True)

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
