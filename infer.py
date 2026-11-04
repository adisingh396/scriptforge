"""ScriptForge inference: turn typed text into a handwritten-looking page image.

Usage:
    python infer.py "Hello world this is ScriptForge" --style 3 --out outputs/page.png
    python infer.py "Same text, five styles" --variants 5 --out outputs/variants.png
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from sf_models import StyleEncoder, UNetGenerator
from sf_render import render_text

DEFAULT_CKPT = "checkpoints/gan_last.pt"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("text", help="text to render as handwriting")
    p.add_argument("--checkpoint", default=DEFAULT_CKPT)
    p.add_argument("--style", type=int, default=0, help="style seed (different seed = different handwriting)")
    p.add_argument("--out", default="outputs/page.png")
    p.add_argument("--variants", type=int, default=0,
                   help="if >0, emit a grid of N style variants of the whole text instead of a page")
    p.add_argument("--scale", type=float, default=3.0, help="page render scale (word height = 48*scale px)")
    p.add_argument("--pdf", action="store_true", help="also save a PDF next to the PNG")
    p.add_argument("--page-width", type=int, default=1600)
    p.add_argument("--page-height", type=int, default=2200)
    p.add_argument("--margin", type=int, default=130)
    p.add_argument("--ink-boost", type=int, default=1, help="1 = normalise each word's ink contrast to pen-like darkness")
    p.add_argument("--thicken", type=int, default=1, help="dilate ink by N px so hollow strokes read as pen pressure (0 = off)")
    p.add_argument("--device", default="")
    return p.parse_args()


def load_generator(ckpt_path, device):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    h, w = int(ck["height"]), int(ck["width"])
    style_dim = int(ck.get("style_dim", 16))
    base = int(ck.get("base", 48))
    G = UNetGenerator(cond_ch=2, style_ch=8, base=base)
    gsd = {k: v for k, v in ck["G_ema"].items() if not k.startswith("_")}
    G.load_state_dict(gsd)  # EMA weights = the inference generator
    G.to(device).eval()
    S = StyleEncoder(z_dim=style_dim)
    if ck.get("S"):
        S.load_state_dict(ck["S"])
    S.to(device).eval()
    # the condition rasterisation must match training exactly
    cargs = ck.get("args") or {}
    from sf_render import find_font
    font_path = cargs.get("cond_font", "cursive")
    if font_path in ("cursive", "block"):
        font_path = find_font(font_path)
    cond_case = cargs.get("cond_case", "keep")
    if cond_case in ("keep", "", None):
        cond_case = None
    return G, S, h, w, style_dim, font_path, cond_case


def generate_word(G, S, text, z, height, width, device, font_path=None, cond_case=None):
    """Generate one word image (uint8 HxW, white bg) with style vector z."""
    glyph = render_text(text, height, width, font_path=font_path, case=cond_case)
    glyph = glyph.astype(np.float32) / 127.5 - 1.0
    g = torch.tensor(glyph, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    g_d = F.max_pool2d(g, 3, stride=1, padding=1)
    cond = torch.cat([g, g_d], dim=1).to(device)
    with torch.no_grad():
        sm = S(z, height, width)
        out = G(cond, sm)
    img = out.float().squeeze().cpu().numpy()
    img = ((img + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    return img


def crop_ink(img, thresh=245, pad=2):
    """Crop to the ink bounding box of a word image (uint8, white bg)."""
    mask = img < thresh
    if not mask.any():
        return img
    ys, xs = np.where(mask)
    y0, y1 = max(0, ys.min() - pad), min(img.shape[0], ys.max() + 1 + pad)
    x0, x1 = max(0, xs.min() - pad), min(img.shape[1], xs.max() + 1 + pad)
    return img[y0:y1, x0:x1]


def boost_contrast(img, target_dark=15.0, strength=0.9, snap=205):
    """Stretch a faint GAN word image to plausible pen contrast with pure-white paper.

    The paper level is taken from the image mode (most pixels are background), then a
    soft floor + snap removes the faint gray halo box the generator leaves behind.
    """
    a = img.astype(np.float32)
    paper = float(np.percentile(a, 75))
    darkest = float(np.percentile(a, 2.0))
    if paper - darkest < 6.0:  # nearly flat image, nothing to stretch
        return img
    span = max(1.0, paper - darkest)
    t = np.clip((paper - a) / span, 0.0, 1.0)     # 0 = paper, 1 = darkest ink
    t = np.clip((t - 0.10) / 0.90, 0.0, 1.0)      # drop the faint halo floor
    t = (1.0 - strength) * t + strength * np.clip(t * 1.2, 0.0, 1.0)
    out = 255.0 - t * (255.0 - target_dark)
    out = np.clip(out, 0, 255).astype(np.uint8)
    if snap:
        out[out >= snap] = 255
    return out


def thicken_ink(img, k=1, blend=0.55):
    """Make hollow/thin generated strokes look like pressure-applied pen ink."""
    from PIL import Image, ImageFilter
    if k <= 0:
        return img
    ink = (255 - img.astype(np.int16)).clip(0, 255).astype(np.uint8)
    thick = np.array(Image.fromarray(ink).filter(ImageFilter.MaxFilter(2 * k + 1)))
    ink2 = (blend * thick.astype(np.float32) + (1.0 - blend) * ink.astype(np.float32))
    out = 255.0 - ink2
    return np.clip(out, 0, 255).astype(np.uint8)


def polish_word(img, thicken=1, smooth=0.7, snap=200, target_dark=15.0):
    """Full post-process for one generated word: contrast -> denoise -> thicken -> snap.

    The generator leaves a faint gray wash and hollow glyph-traced strokes; this turns
    them into solid dark ink on pure white paper without touching letter shapes.
    """
    from PIL import Image, ImageFilter
    out = boost_contrast(img, target_dark=target_dark, snap=0)
    if smooth and smooth > 0:
        out = np.array(Image.fromarray(out).filter(ImageFilter.GaussianBlur(smooth)))
    if thicken and thicken > 0:
        out = thicken_ink(out, int(thicken))
    if snap:
        out[out >= snap] = 255
    return out


def compose_page(G, S, text, style_seed, args, height, width, device,
                 font_path=None, cond_case=None):
    """Wrap text into lines and paste generated words onto a page canvas."""
    rng = np.random.default_rng(style_seed)
    gen = torch.Generator().manual_seed(style_seed)
    z = torch.randn(1, S.fc.in_features, generator=gen).to(device)

    scale = max(1.0, float(args.scale))
    word_h = int(height * scale)
    page = np.full((args.page_height, args.page_width), 255, dtype=np.uint8)
    margin = args.margin
    max_w = args.page_width - 2 * margin

    words = [w for w in text.upper().split() if w]
    if not words:
        raise ValueError("no renderable words in input (allowed: A-Z and spaces)")

    x, y = margin, margin
    line = 0
    space_w = int(word_h * 0.38)
    line_h = int(word_h * 1.65)

    for word in words:
        img = generate_word(G, S, word, z, height, width, device,
                            font_path=font_path, cond_case=cond_case)
        if args.ink_boost:
            img = polish_word(img, thicken=getattr(args, "thicken", 1))
        crop = crop_ink(img)
        if crop.shape[0] < 1:
            continue
        # scale word to target height, keep aspect
        s = word_h / crop.shape[0]
        w2 = max(1, int(round(crop.shape[1] * s)))
        if w2 > max_w:  # ultra-long word: squeeze to fit the line
            w2 = max_w
        from PIL import Image
        wimg = np.array(Image.fromarray(crop).resize((w2, word_h), Image.LANCZOS))

        # slight per-word rotation so the page does not look mechanically stamped
        angle = float(rng.uniform(-1.3, 1.3))
        if abs(angle) > 0.15:
            rot = Image.fromarray(wimg).rotate(angle, resample=Image.BICUBIC,
                                               expand=True, fillcolor=255)
            wimg = np.array(rot)
        w2, wh = wimg.shape[1], wimg.shape[0]

        if x + w2 > margin + max_w:
            x = margin
            line += 1
            y = margin + line * line_h
            if y + wh > args.page_height:
                break  # page full

        # per-word baseline jitter, deterministic from (seed, word index)
        jitter = int(rng.integers(-word_h // 12, word_h // 12 + 1))
        y_off = max(0, min(args.page_height - wh, y + jitter))
        region = page[y_off:y_off + wh, x:x + w2]
        page[y_off:y_off + wh, x:x + w2] = np.minimum(region, wimg[:region.shape[0], :region.shape[1]])
        x += w2 + space_w

    return page


def variants_grid(G, S, text, args, height, width, device,
                  font_path=None, cond_case=None):
    """Grid of N style variants of one line of text."""
    from PIL import Image
    n = max(1, min(args.variants, 8))
    words = [w for w in text.upper().split() if w][:6]
    rows = []
    cell_h = height
    for k in range(n):
        seed = args.style + 1000 * k
        gen = torch.Generator().manual_seed(seed)
        z = torch.randn(1, S.fc.in_features, generator=gen).to(device)
        imgs = [crop_ink(polish_word(generate_word(G, S, w, z, height, width, device,
                                                            font_path=font_path,
                                                            cond_case=cond_case)))
                for w in words]
        target_h = cell_h
        scaled = []
        for im in imgs:
            s = target_h / im.shape[0]
            scaled.append(np.array(Image.fromarray(im).resize(
                (max(1, int(im.shape[1] * s)), target_h), Image.LANCZOS)))
        gap = 14
        row_w = sum(im.shape[1] for im in scaled) + gap * (len(scaled) - 1)
        row = np.full((target_h, row_w), 255, dtype=np.uint8)
        xx = 0
        for im in scaled:
            row[:, xx:xx + im.shape[1]] = np.minimum(row[:, xx:xx + im.shape[1]], im)
            xx += im.shape[1] + gap
        rows.append(row)
    row_w = max(r.shape[1] for r in rows)
    gap_rows = 10
    grid = np.full((len(rows) * cell_h + gap_rows * (len(rows) - 1), row_w), 255, dtype=np.uint8)
    yy = 0
    for r in rows:
        grid[yy:yy + r.shape[0], :r.shape[1]] = np.minimum(grid[yy:yy + r.shape[0], :r.shape[1]], r)
        yy += r.shape[0] + gap_rows
    return grid


def main():
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    G, S, height, width, style_dim, font_path, cond_case = load_generator(args.checkpoint, device)
    print(f"loaded {args.checkpoint} | canvas {height}x{width} | device {device}", flush=True)

    os.makedirs(os.path.dirname(args.out) or "outputs", exist_ok=True)
    if args.variants and args.variants > 0:
        img = variants_grid(G, S, args.text, args, height, width, device,
                            font_path=font_path, cond_case=cond_case)
    else:
        img = compose_page(G, S, args.text, args.style, args, height, width, device,
                           font_path=font_path, cond_case=cond_case)

    from PIL import Image
    Image.fromarray(img).save(args.out)
    print(f"saved -> {args.out}", flush=True)
    if args.pdf:
        pdf_path = os.path.splitext(args.out)[0] + ".pdf"
        Image.fromarray(img).convert("L").save(pdf_path, "PDF", resolution=150)
        print(f"saved -> {pdf_path}", flush=True)


if __name__ == "__main__":
    main()
