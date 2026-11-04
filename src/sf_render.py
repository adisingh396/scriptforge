"""Deterministic, torch-free text rasterizer producing normalized glyph images."""

import functools
import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Font auto-discovery: cursive/handwriting faces first (the condition must look like
# connected script, otherwise the generator is instructed to draw block capitals).
_CURSIVE_CANDIDATES = [
    r"C:\Windows\Fonts\segoesc.ttf",     # Segoe Script
    r"C:\Windows\Fonts\segoescb.ttf",    # Segoe Script Bold
    r"C:\Windows\Fonts\LHANDW.TTF",      # Lucida Handwriting
    r"C:\Windows\Fonts\Inkfree.ttf",     # Ink Free
    r"C:\Windows\Fonts\BRADHITC.TTF",    # Bradley Hand ITC
    r"C:\Windows\Fonts\ITCKRIST.TTF",    # Kristan ITC
    r"C:\Windows\Fonts\Gabriola.ttf",    # Gabriola (calligraphic)
]

_BLOCK_CANDIDATES = [
    r"C:\Windows\Fonts\arial.ttf",
    r"C:\Windows\Fonts\seguisb.ttf",
    r"C:\Windows\Fonts\calibri.ttf",
    r"C:\Windows\Fonts\comic.ttf",
]


def find_font(kind="cursive"):
    """Return the best available font path for 'cursive' or 'block'."""
    cands = _CURSIVE_CANDIDATES if kind == "cursive" else _BLOCK_CANDIDATES
    for p in cands:
        if os.path.exists(p):
            return p
    return None


def _resolve_default_font():
    return find_font("cursive") or find_font("block")


FONT_PATH = _resolve_default_font()
print(f"sf_render: resolved font = {FONT_PATH if FONT_PATH else 'PIL default'}")


@functools.lru_cache(maxsize=64)
def _load_font(size, font_path=None):
    """Cache font objects: parsing a TTF from disk costs ~10-30 ms per call."""
    path = font_path if font_path else FONT_PATH
    if path and os.path.exists(path):
        try:
            return ImageFont.truetype(path, max(1, int(size)))
        except Exception:
            pass
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _measure(font, text):
    bbox = font.getbbox(text)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    return w, h, bbox


def _render_text_impl(text, height, width, font_path=None):
    """Render `text` into a (height, width) uint8 glyph image.

    White background (255), black ink (0). Text is drawn exactly as given (case and
    spacing come from the caller) so cursive conditions keep their joins.
    """
    height = int(height)
    width = int(width)
    canvas = np.full((height, width), 255, dtype=np.uint8)

    text = "" if text is None else str(text)
    if text.strip() == "":
        return canvas

    # Auto-fit.
    size = int(0.62 * height)
    font = _load_font(size, font_path)
    tw, th, bbox = _measure(font, text)

    # Scale (a few iterations for stability).
    for _ in range(4):
        if tw <= 0 or th <= 0:
            break
        scale_w = (width * 0.92) / max(1, tw)
        scale_h = (height * 0.75) / max(1, th)
        scale = min(scale_w, scale_h)
        new_size = int(size * scale)
        if new_size < 1:
            new_size = 1
        if new_size == size:
            break
        size = new_size
        font = _load_font(size, font_path)
        tw, th, bbox = _measure(font, text)

    tw, th, bbox = _measure(font, text)
    x = (width - tw) // 2 - bbox[0]
    y = (height - th) // 2 - bbox[1]

    img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(img)
    draw.text((x, y), text, font=font, fill=0)
    return np.array(img, dtype=np.uint8)


@functools.lru_cache(maxsize=8192)
def _render_cached(text, height, width, font_path):
    return _render_text_impl(text, height, width, font_path)


def apply_case(text, case="title"):
    """case: 'upper' | 'lower' | 'title' (Title keeps the capital initial of names)."""
    s = str(text)
    if case == "upper":
        return s.upper()
    if case == "lower":
        return s.lower()
    if case == "title":
        return " ".join(w[:1].upper() + w[1:].lower() for w in s.split(" "))
    return s


def render_text(text, height, width, font_path=None, case=None):
    """Render `text` into a (height, width) uint8 glyph image (cached)."""
    s = "" if text is None else str(text)
    if case:
        s = apply_case(s, case)
    arr = _render_cached(s, int(height), int(width), font_path)
    return arr.copy()


def render_batch(texts, height=48, width=192, font_path=None, case=None):
    out = np.zeros((len(texts), int(height), int(width)), dtype=np.uint8)
    for i, t in enumerate(texts):
        out[i] = render_text(t, height, width, font_path=font_path, case=case)
    return out
