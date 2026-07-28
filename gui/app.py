"""
Interactive Streamlit front-end for the fractal codec.

Pure UI wiring around the existing public API (fractal_compression/) --
no encoding/decoding logic lives here. Run with:

    streamlit run gui/app.py
"""
from __future__ import annotations
import io
import math
import os
import sys
import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import streamlit as st
from PIL import Image
import pillow_heif
from skimage import data as skdata

pillow_heif.register_heif_opener()  # lets Image.open() read HEIC/HEIF (iPhone default) too

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fractal_compression import (
    FractalConfig, encode_channel, decode_channel, encode_image, decode_image,
)
from fractal_compression.entropy import encode_channel_entropy
from fractal_compression.metrics import psnr, ssim_score

SAMPLE_IMAGES = {
    "camera": skdata.camera,
    "coins": skdata.coins,
    "astronaut": skdata.astronaut,
    "chelsea": skdata.chelsea,
}
RESOLUTIONS = [64, 96, 128, 160, 192, 256]
ERROR_THRESHOLDS = [10, 20, 30, 60, 100, 150, 250, 400, 700, 1200, 2000, 5000]
QUADTREE_COLOR = "#00c2c2"

st.set_page_config(page_title="Fractal Image Compression", layout="wide")


def load_source_image(source, uploaded_file, sample_name, color_mode) -> Image.Image | None:
    mode = "RGB" if color_mode == "Color" else "L"
    if source == "Upload":
        if uploaded_file is None:
            return None
        # Read the raw bytes explicitly rather than handing Image.open the
        # UploadedFile stream directly -- its read position isn't always at
        # byte 0 by the time it reaches here, which PIL reports as an
        # UnidentifiedImageError even for a perfectly valid file.
        return Image.open(io.BytesIO(uploaded_file.getvalue())).convert(mode)
    return Image.fromarray(SAMPLE_IMAGES[sample_name]()).convert(mode)


def resize_preserve_aspect(pil_img: Image.Image, max_dim: int) -> Image.Image:
    img = pil_img.copy()
    img.thumbnail((max_dim, max_dim), Image.LANCZOS)
    return img


def to_display(arr: np.ndarray) -> np.ndarray:
    return np.clip(arr, 0, 255).astype(np.uint8)


def channel_bytes(channel_enc, use_entropy: bool) -> bytes:
    return encode_channel_entropy(channel_enc) if use_entropy else channel_enc.to_bytes()


def show_image(col, title, arr, is_color, overlay_transforms=None):
    with col:
        st.subheader(title)
        fig, ax = plt.subplots()
        ax.imshow(to_display(arr), cmap=None if is_color else "gray", vmin=0, vmax=255)
        if overlay_transforms is not None:
            for t in overlay_transforms:
                ax.add_patch(patches.Rectangle(
                    (t.col, t.row), t.cols, t.rows,
                    linewidth=0.6, edgecolor=QUADTREE_COLOR, facecolor="none",
                ))
        ax.axis("off")
        st.pyplot(fig, clear_figure=True)


# ---------------------------------------------------------------- sidebar --
with st.sidebar:
    st.header("Image")
    source = st.radio("Source", ["Sample image", "Upload"], index=0)
    if source == "Upload":
        uploaded_file = st.file_uploader(
            "Upload an image", type=["png", "jpg", "jpeg", "bmp", "heic", "heif"])
        sample_name = None
    else:
        uploaded_file = None
        sample_name = st.selectbox("Sample", list(SAMPLE_IMAGES.keys()))

    color_mode = st.radio("Color mode", ["Grayscale", "Color"], index=0)
    chroma_subsample = False
    if color_mode == "Color":
        chroma_subsample = st.checkbox("4:2:0 chroma subsampling", value=True)

    resolution = st.select_slider("Working resolution (longer side, px)", options=RESOLUTIONS, value=128)

    st.header("Quadtree")
    error_thresh = st.select_slider("Error threshold", options=ERROR_THRESHOLDS, value=100)
    quantization_aware = st.checkbox(
        "Quantization-aware domain search", value=False,
        help="Picks each block's domain position by post-quantization error "
             "instead of continuous error — same cost, but measured up to "
             "+12 dB PSNR on some images. See the README for why.",
    )

    with st.expander("Advanced settings"):
        max_block = st.selectbox("max_block", [16, 32, 64, 128], index=2)
        min_block = st.selectbox("min_block", [v for v in [2, 4, 8, 16] if v <= max_block], index=0)
        step = st.select_slider("search step (stride)", options=[1, 2, 4], value=1)
        k_bits = st.slider("k_bits", 3, 8, 5)
        c_bits = st.slider("c_bits", 4, 8, 7)

    entropy_on = st.checkbox("Also entropy-code the bitstream (Huffman)", value=False)

    if resolution >= 192 and step == 1 and color_mode == "Color":
        st.caption(
            "⚠️ Large resolution + step=1 + color can be slow — this repo's own "
            "benchmarks saw ~2s+ per channel at just 96×96."
        )

    encode_clicked = st.button("Encode image", type="primary")

# ---------------------------------------------------------------- header --
st.title("Fractal Image Compression — Interactive Encoder")
st.caption(
    "Upload an image, tune the causal quadtree codec, and see the reconstruction, "
    "its quadtree partition, and its real rate/quality numbers."
)

# ---------------------------------------------------------------- encode --
if encode_clicked:
    pil_img, load_error = None, False
    try:
        pil_img = load_source_image(source, uploaded_file, sample_name, color_mode)
    except Exception as e:
        load_error = True
        st.sidebar.error(
            f"Couldn't read that file as an image ({type(e).__name__}: {e}). "
            "If it's an iPhone photo, HEIC is supported, but a re-exported "
            "PNG/JPEG is worth trying if this persists."
        )

    if load_error:
        pass
    elif pil_img is None:
        st.sidebar.error("Please upload an image first.")
    else:
        pil_img = resize_preserve_aspect(pil_img, resolution)
        arr = np.array(pil_img).astype(np.float64)
        cfg = FractalConfig(error_thresh=error_thresh, max_block=max_block, min_block=min_block,
                             step=step, k_bits=k_bits, c_bits=c_bits,
                             quantization_aware=quantization_aware)
        is_color = color_mode == "Color"

        with st.spinner("Encoding…"):
            t0 = time.time()
            if is_color:
                enc = encode_image(arr, cfg, chroma_subsample=chroma_subsample)
                enc_time = time.time() - t0
                t0 = time.time()
                recon = decode_image(enc)
                dec_time = time.time() - t0
                recon = np.clip(recon, 0, 255)

                channels = enc.channels
                leaves = sum(len(c.transforms) for c in channels)
                overlay_transforms = channels[0].transforms  # Y channel, full resolution
                quality_ssim = ssim_score(arr.astype(np.uint8), recon.astype(np.uint8))
            else:
                enc = encode_channel(arr, cfg)
                enc_time = time.time() - t0
                t0 = time.time()
                recon = decode_channel(enc)
                dec_time = time.time() - t0
                recon = np.clip(recon, 0, 255)

                channels = [enc]
                leaves = len(enc.transforms)
                overlay_transforms = enc.transforms
                quality_ssim = ssim_score(arr, recon)

            fixed_streams = [(l, channel_bytes(c, False)) for l, c in
                              zip(["Y", "Cb", "Cr"] if is_color else ["gray"], channels)]
            bpp_fixed = sum(len(b) for _, b in fixed_streams) * 8 / (arr.shape[0] * arr.shape[1])

            entropy_streams, bpp_entropy = None, None
            if entropy_on:
                entropy_streams = [(l, channel_bytes(c, True)) for l, c in
                                    zip(["Y", "Cb", "Cr"] if is_color else ["gray"], channels)]
                bpp_entropy = sum(len(b) for _, b in entropy_streams) * 8 / (arr.shape[0] * arr.shape[1])

        st.session_state["result"] = dict(
            original=arr, recon=recon, is_color=is_color, transforms=overlay_transforms,
            bpp_fixed=bpp_fixed, bpp_entropy=bpp_entropy,
            leaves=leaves, enc_time=enc_time, dec_time=dec_time,
            psnr=psnr(arr, recon), ssim=quality_ssim,
            padded_size=channels[0].padded_size,
            fixed_streams=fixed_streams, entropy_streams=entropy_streams,
        )

# ---------------------------------------------------------------- output --
result = st.session_state.get("result")

if result is None:
    st.info("Pick an image source and settings in the sidebar, then click **Encode image**.")
else:
    col1, col2, col3 = st.columns(3)
    show_image(col1, "Original", result["original"], result["is_color"])
    show_image(col2, "Reconstructed", result["recon"], result["is_color"])
    show_image(col3, "Quadtree partition", result["recon"], result["is_color"], result["transforms"])

    st.divider()
    m1, m2, m3, m4, m5, m6, m7 = st.columns(7)
    psnr_val = result["psnr"]
    m1.metric("PSNR", "∞ dB" if math.isinf(psnr_val) else f"{psnr_val:.2f} dB")
    m2.metric("SSIM", f"{result['ssim']:.4f}")
    m3.metric("bpp (fixed)", f"{result['bpp_fixed']:.3f}")
    m4.metric("bpp (entropy)", f"{result['bpp_entropy']:.3f}" if result["bpp_entropy"] is not None else "—")
    m5.metric("Leaves", result["leaves"])
    m6.metric("Encode time", f"{result['enc_time']:.2f}s")
    m7.metric("Decode time", f"{result['dec_time'] * 1000:.1f}ms")

    h, w = result["original"].shape[:2]
    st.caption(
        f"Resized to {w}×{h} px, encoded on an internal padded canvas of "
        f"{result['padded_size']}×{result['padded_size']} (padding is transparent — "
        f"the decoder crops back to {w}×{h})."
    )

    st.divider()
    dl1, dl2 = st.columns(2)
    with dl1:
        recon_img = Image.fromarray(to_display(result["recon"]))
        buf = io.BytesIO()
        recon_img.save(buf, format="PNG")
        st.download_button("Download reconstructed PNG", buf.getvalue(),
                            file_name="reconstructed.png", mime="image/png")
    with dl2:
        fixed_blob = b"".join(b for _, b in result["fixed_streams"])
        st.download_button("Download bitstream (fixed-width)", fixed_blob,
                            file_name="fractal_fixed.bin", mime="application/octet-stream")
        if result["entropy_streams"] is not None:
            entropy_blob = b"".join(b for _, b in result["entropy_streams"])
            st.download_button("Download bitstream (entropy-coded)", entropy_blob,
                                file_name="fractal_entropy.bin", mime="application/octet-stream")
    st.caption(
        "Color downloads are each channel's real serialized bytes concatenated "
        "(Y, then Cb, then Cr) for convenience — not a defined multi-channel "
        "container format."
    )
