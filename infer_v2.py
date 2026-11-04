"""ScriptForge v2 inference: render a requested string to a handwriting PNG.

Usage:
    python infer_v2.py --text "Simon" --n 2 --out outputs/infer_v2/simon.png
"""

import argparse
import os
import re
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from sf_data import MODEL_CHARSET, PAD_ID, text_to_ids
from sf_models_v2 import TextEncoder, TextCondition, UNetGeneratorV2


class GenStack(torch.nn.Module):
    """text encoder + conditioner + U-Net as one module (mirrors training)."""

    def __init__(self, T, C, G):
        super().__init__()
        self.T, self.C, self.G = T, C, G

    def forward(self, ids, lens, z=None, writer=None):
        enc, mask = self.T(ids, lens)
        cond = self.C(enc, mask, z, writer, lens)
        return self.G(cond, z)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", default="checkpoints/gan_v2.pt")
    p.add_argument("--text", default="hello world")
    p.add_argument("--out", default="")
    p.add_argument("--style-seed", type=int, default=0, dest="style_seed")
    p.add_argument("--writer", default="", help="writer id (e.g. a01) or int, for writer-conditioned checkpoints")
    p.add_argument("--n", type=int, default=1)
    p.add_argument("--device", default="")
    p.add_argument("--ema", type=int, default=1)
    return p.parse_args()


def main():
    args = parse_args()

    ids_list = text_to_ids(args.text)
    if not ids_list:
        print(f"error: text has no characters representable in MODEL_CHARSET "
              f"({MODEL_CHARSET!r})", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(args.ckpt):
        print(f"error: checkpoint not found: {args.ckpt}", file=sys.stderr)
        sys.exit(1)

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)

    # arch gate
    if "arch" in ck:
        if ck["arch"] != "v2":
            print(f"error: {args.ckpt} is a v1 checkpoint "
                  f"(arch={ck['arch']!r}); use infer.py instead", file=sys.stderr)
            sys.exit(1)
    else:
        if not all(k in ck for k in ("T", "C", "G", "G_stack_ema")):
            print(f"error: {args.ckpt} is a v1 checkpoint "
                  f"(missing v2 keys); use infer.py instead", file=sys.stderr)
            sys.exit(1)

    height = int(ck["height"])
    width = int(ck["width"])
    base = int(ck["base"])
    cond_ch = int(ck["cond_ch"])
    style_dim = int(ck["style_dim"])
    text_embed = int(ck["text_embed"])
    text_hidden = int(ck["text_hidden"])
    vocab = int(ck["vocab"])
    writer_cond = bool(ck.get("writer_cond", False))
    n_writers = int(ck.get("n_writers", 0) or 0)
    writers = ck.get("writers", []) if isinstance(ck.get("writers"), list) else []

    T_enc = TextEncoder(vocab, embed_dim=text_embed, hidden=text_hidden)
    C = TextCondition(T_enc.out_dim, cond_ch=cond_ch, height=height, width=width,
                      style_dim=style_dim, n_writers=n_writers, writer_cond=writer_cond)
    G = UNetGeneratorV2(cond_ch=cond_ch, base=base)

    if args.ema:
        state = dict(ck["G_stack_ema"])
        state.pop("_num_updates", None)
        stack = GenStack(T_enc, C, G)
        stack.load_state_dict(state, strict=True)
    else:
        T_enc.load_state_dict(ck["T"], strict=True)
        C.load_state_dict(ck["C"], strict=True)
        G.load_state_dict(ck["G"], strict=True)
        stack = GenStack(T_enc, C, G)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    stack.to(device).eval()

    def _writer_idx():
        if not writer_cond:
            return None
        s = (args.writer or "").strip().lower()
        if not s:
            return 0
        if s.isdigit():
            return int(s)
        if writers:
            known = [w.lower() for w in writers]
            if s in known:
                return known.index(s)
            return 0
        return 0

    writer_idx = _writer_idx()
    # warm-up
    L = len(ids_list)
    ids_t = torch.full((1, L), PAD_ID, dtype=torch.long)
    ids_t[0, :L] = torch.tensor(ids_list, dtype=torch.long)
    lens_t = torch.tensor([L], dtype=torch.long)
    w_t = torch.tensor([writer_idx], dtype=torch.long) if writer_cond else None
    z_dummy = (torch.zeros(1, style_dim, dtype=torch.float32)
               if style_dim > 0 and not writer_cond else None)
    with torch.inference_mode():
        _ = stack(ids_t.to(device), lens_t.to(device),
                  z_dummy.to(device) if z_dummy is not None else None,
                  w_t.to(device) if w_t is not None else None)

    # render
    gen = (torch.Generator(device="cpu").manual_seed(args.style_seed)
           if style_dim > 0 and not writer_cond else None)
    rows = []
    for _ in range(max(1, args.n)):
        if style_dim > 0 and writer_cond:
            z = None  # style comes from the writer embedding
        elif style_dim > 0:
            z = torch.randn(1, style_dim, generator=gen)
        else:
            z = None
        with torch.inference_mode():
            out = stack(ids_t.to(device), lens_t.to(device),
                        z.to(device) if z is not None else None,
                        w_t.to(device) if w_t is not None else None)
        img = out.float().squeeze(0).squeeze(0).cpu().numpy()
        img = np.clip((img + 1.0) * 127.5, 0, 255).astype(np.uint8)
        rows.append(img)

    grid = np.concatenate(rows, axis=0)

    out_path = args.out
    if not out_path:
        slug = re.sub(r"[^a-z0-9]", "_", args.text.lower())[:40]
        out_path = os.path.join("outputs", "infer_v2", f"{slug}_{args.style_seed}.png")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    from PIL import Image
    Image.fromarray(grid).save(out_path)

    print(f"saved -> {out_path}", flush=True)
    print(f"checkpoint step: {ck['step']}", flush=True)


if __name__ == "__main__":
    main()
