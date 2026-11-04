"""ScriptForge web app: type text, get it written as handwriting.

Run:  streamlit run app.py
"""

import argparse
import os
import sys

import numpy as np
import streamlit as st
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from infer import compose_page, load_generator

CKPT = os.environ.get("SCRIPTFORGE_CKPT", "checkpoints/gan_iam.pt")


@st.cache_resource(show_spinner="loading generator...")
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return load_generator(CKPT, device) + (device,)


def main():
    st.set_page_config(page_title="ScriptForge", page_icon="✍️", layout="centered")
    st.title("ScriptForge")
    st.caption("Type anything — the GAN writes it as handwriting learned from real handwritten names.")

    if not os.path.exists(CKPT):
        st.error(f"Generator checkpoint not found: `{CKPT}`.\n\nTrain it first:\n\n"
                 "```\npython train_gan.py --data-dir data/prepared --steps 30000\n```")
        return

    G, S, height, width, style_dim, font_path, cond_case, device = load_models()
    st.success(f"Loaded generator ({height}x{width} canvas) on **{device.type}**")

    text = st.text_area("Text to write", "Hello world this is ScriptForge writing", height=120)

    col1, col2, col3 = st.columns(3)
    with col1:
        style = st.number_input("Style seed", min_value=0, max_value=9999, value=7,
                                help="Different seeds give different handwriting styles")
    with col2:
        scale = st.slider("Text size", 1.0, 6.0, 3.0, 0.5)
    with col3:
        st.write("")  # spacing
        generate = st.button("✍️ Generate handwriting", type="primary")

    if generate:
        if not text.strip():
            st.warning("Type some text first.")
            return
        args = argparse.Namespace(
            scale=scale, page_width=1600, page_height=2200, margin=130, ink_boost=1)
        with st.spinner("writing..."):
            img = compose_page(G, S, text, int(style), args, height, width, device,
                               font_path=font_path, cond_case=cond_case)
        st.image(img, use_container_width=True, clamp=True)

        from PIL import Image
        im = Image.fromarray(img)
        st.download_button("⬇️ Download PNG", data=_png_bytes(im),
                           file_name=f"scriptforge_style{style}.png", mime="image/png")
        st.download_button("⬇️ Download PDF", data=_pdf_bytes(im),
                           file_name=f"scriptforge_style{style}.pdf", mime="application/pdf")


def _png_bytes(im):
    import io
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def _pdf_bytes(im):
    import io
    buf = io.BytesIO()
    im.convert("L").save(buf, format="PDF", resolution=150)
    return buf.getvalue()


if __name__ == "__main__":
    main()
