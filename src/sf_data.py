"""Word-level dataset + loaders for ScriptForge handwriting GAN."""

import os
import sys
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance

# Make sf_render importable when running from src/ or scripts/.
try:
    from sf_render import render_text
except ImportError:
    _here = os.path.dirname(os.path.abspath(__file__))
    if _here not in sys.path:
        sys.path.insert(0, _here)
    from sf_render import render_text

# Lazy torch import.
try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    _TORCH = True
except Exception:
    torch = None
    Dataset = object
    DataLoader = None
    _TORCH = False


def _require_torch():
    if not _TORCH:
        raise RuntimeError("torch is not installed; cannot use WordDataset.")


# ---------- dilation ----------

def dilate_glyph(glyph_uint8, k=3):
    """Dilate ink in a (H,W) uint8 glyph using a kxk max-filter.

    Ink is black (0); we invert so ink becomes high, max-pool, then invert back.
    Returns uint8 (H,W).
    """
    _require_torch()
    import torch.nn.functional as F
    t = torch.from_numpy(np.asarray(glyph_uint8, dtype=np.uint8)).float()
    t = t.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    inv = 255.0 - t
    pad = k // 2
    dil = F.max_pool2d(inv, kernel_size=k, stride=1, padding=pad)
    out = 255.0 - dil
    return out.squeeze(0).squeeze(0).clamp(0, 255).to(torch.uint8).numpy()


def _erode_glyph(glyph_uint8, k=3):
    _require_torch()
    import torch.nn.functional as F
    t = torch.from_numpy(np.asarray(glyph_uint8, dtype=np.uint8)).float()
    t = t.unsqueeze(0).unsqueeze(0)
    # erosion of ink (black) = max filter on original (shrink black)
    er = F.max_pool2d(t, kernel_size=k, stride=1, padding=k // 2)
    return er.squeeze(0).squeeze(0).clamp(0, 255).to(torch.uint8).numpy()


# ---------- augmentation ----------

def _augment_image(img_uint8):
    """Augment a (H,W) uint8 grayscale image. Returns uint8 (H,W)."""
    img = Image.fromarray(np.asarray(img_uint8, dtype=np.uint8))
    H, W = img_uint8.shape

    # random rotation +-2 deg
    if np.random.rand() < 0.5:
        ang = float(np.random.uniform(-2.0, 2.0))
        img = img.rotate(ang, resample=Image.BILINEAR, fillcolor=255)

    # random dilation/erosion pick
    r = np.random.rand()
    if r < 0.33:
        img = img.filter(ImageFilter.MinFilter(3))   # dilate ink (black)
    elif r < 0.66:
        img = img.filter(ImageFilter.MaxFilter(3))   # erode ink

    # brightness/contrast
    if np.random.rand() < 0.5:
        img = ImageEnhance.Brightness(img).enhance(float(np.random.uniform(0.9, 1.1)))
    if np.random.rand() < 0.5:
        img = ImageEnhance.Contrast(img).enhance(float(np.random.uniform(0.9, 1.1)))

    # horizontal shift <= 3 px
    shift = int(np.random.randint(-3, 4))
    if shift != 0:
        arr = np.array(img)
        out = np.full_like(arr, 255)
        if shift > 0:
            out[:, shift:] = arr[:, :-shift]
        else:
            out[:, :shift] = arr[:, -shift:]
        img = Image.fromarray(out)

    out = np.array(img, dtype=np.uint8)
    # keep ink polarity sane: never invert
    return out


# ---------- text condition (v2) ----------

# The v2 generator is conditioned on character ids instead of a font raster. Charset must match
# src/sf_models.py exactly; importing it keeps the two from drifting apart.
try:
    try:
        from sf_models import CHARSET as MODEL_CHARSET
    except ImportError:
        import os as _os
        import sys as _sys
        _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
        from sf_models import CHARSET as MODEL_CHARSET
except Exception:  # torch missing — still allow imports to succeed
    MODEL_CHARSET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ "

PAD_ID = len(MODEL_CHARSET)   # one extra embedding row used only for padding


def text_to_ids(text, charset=MODEL_CHARSET):
    idx = {c: i for i, c in enumerate(charset)}
    return [idx[c] for c in str(text).upper() if c in idx]


def collate_text(batch):
    """(real, ids, lengths, texts) with ids right-padded by PAD_ID."""
    reals = torch.stack([b["real"] for b in batch])
    seqs = [b["ids"] for b in batch]
    lengths = torch.tensor([max(1, len(s)) for s in seqs], dtype=torch.long)
    L = max(1, max(len(s) for s in seqs))
    ids = torch.full((len(seqs), L), PAD_ID, dtype=torch.long)
    for i, s in enumerate(seqs):
        if s:
            ids[i, :len(s)] = torch.tensor(s, dtype=torch.long)
        else:
            ids[i, 0] = PAD_ID
    writers = None
    if "writer" in batch[0]:
        writers = torch.tensor([int(b["writer"]) for b in batch], dtype=torch.long)
    return reals, ids, lengths, [b["text"] for b in batch], writers


# ---------- dataset ----------

class WordDataset(Dataset):
    def __init__(self, imgs_path, texts_path, height=None, width=None, augment=False,
                 font_path=None, font_case=None, cond_mode="glyph", writers_path=None,
                 n_writers=0):
        _require_torch()
        self.imgs = np.load(imgs_path, mmap_mode="r")
        with open(texts_path, "r", encoding="utf-8") as f:
            self.texts = [line.rstrip("\n") for line in f]
        if len(self.imgs) != len(self.texts):
            raise ValueError(f"imgs ({len(self.imgs)}) vs texts ({len(self.texts)}) count mismatch")
        self.writers = None
        if writers_path is not None:
            w = np.load(writers_path)
            if len(w) != len(self.imgs):
                raise ValueError(f"writers ({len(w)}) vs imgs ({len(self.imgs)}) count mismatch")
            self.writers = w
            self.n_writers = max(int(n_writers) or 0, int(w.max()) + 1)
        self.height = int(height) if height is not None else int(self.imgs.shape[1])
        self.width = int(width) if width is not None else int(self.imgs.shape[2])
        self.augment = bool(augment)
        # condition rasterisation: cursive font + real case keeps letters joined
        self.font_path = font_path
        self.font_case = font_case
        # "glyph" = v1 (font raster + dilation); "text" = v2 (character ids only)
        self.cond_mode = cond_mode
        if cond_mode not in ("glyph", "text"):
            raise ValueError(f"cond_mode must be 'glyph' or 'text', got {cond_mode!r}")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        _require_torch()
        img = np.array(self.imgs[idx], dtype=np.uint8).copy()
        if self.augment:
            img = _augment_image(img)

        text = self.texts[idx]
        real = torch.from_numpy(img).float().unsqueeze(0)  # (1,H,W)
        real = real / 127.5 - 1.0

        if self.cond_mode == "text":
            # v2: no font raster; the network reads the characters themselves
            out = {"real": real, "ids": text_to_ids(text), "text": text}
            if self.writers is not None:
                out["writer"] = int(self.writers[idx])
            return out

        glyph = render_text(text, self.height, self.width,
                            font_path=self.font_path, case=self.font_case)
        dilated = dilate_glyph(glyph, 3)

        cond_arr = np.stack([glyph, dilated], axis=0).astype(np.uint8)  # (2,H,W)
        cond = torch.from_numpy(cond_arr).float()
        cond = cond / 127.5 - 1.0

        return {"real": real, "cond": cond, "ids": text_to_ids(text), "text": text}


def _cfg_get(cfg, key, default=None):
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    if hasattr(cfg, key):
        return getattr(cfg, key)
    return default


def build_loaders(cfg):
    _require_torch()
    data_dir = _cfg_get(cfg, "data_dir", "data/prepared")
    height = int(_cfg_get(cfg, "height", 48))
    width = int(_cfg_get(cfg, "width", 192))
    batch_size = int(_cfg_get(cfg, "batch_size", _cfg_get(cfg, "batch", 16)))
    num_workers = int(_cfg_get(cfg, "num_workers", 0))
    pin_memory = bool(_cfg_get(cfg, "pin_memory", True))

    train_imgs = os.path.join(data_dir, "train_imgs.npy")
    train_texts = os.path.join(data_dir, "train_texts.txt")
    val_imgs = os.path.join(data_dir, "validation_imgs.npy")
    val_texts = os.path.join(data_dir, "validation_texts.txt")

    train_ds = WordDataset(train_imgs, train_texts, height, width, augment=True,
                           font_path=_cfg_get(cfg, "font_path", None),
                           font_case=_cfg_get(cfg, "font_case", None))
    val_ds = WordDataset(val_imgs, val_texts, height, width, augment=False,
                         font_path=_cfg_get(cfg, "font_path", None),
                         font_case=_cfg_get(cfg, "font_case", None))

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    return train_loader, val_loader


def list_datasets(data_dir="data/prepared"):
    splits = []
    for s in ("train", "validation", "test"):
        if os.path.exists(os.path.join(data_dir, f"{s}_imgs.npy")):
            splits.append(s)
    return splits
