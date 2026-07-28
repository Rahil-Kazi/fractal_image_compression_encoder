"""JPEG baseline (via Pillow) for rate-distortion comparison plots."""
from __future__ import annotations
import io
import numpy as np
from PIL import Image


def jpeg_encode_decode(img: np.ndarray, quality: int):
    """Returns (reconstructed_array, n_bytes)."""
    is_color = img.ndim == 3
    mode = "RGB" if is_color else "L"
    pil_img = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), mode=mode)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality)
    n_bytes = buf.tell()
    buf.seek(0)
    decoded = np.array(Image.open(buf).convert(mode)).astype(np.float64)
    return decoded, n_bytes


def jpeg_rate_distortion(img: np.ndarray, qualities=(10, 20, 30, 40, 50, 60, 70, 80, 90)):
    """Returns list of dicts: quality, bpp, n_bytes."""
    h, w = img.shape[:2]
    n_channels = img.shape[2] if img.ndim == 3 else 1
    results = []
    for q in qualities:
        decoded, n_bytes = jpeg_encode_decode(img, q)
        bpp = (n_bytes * 8) / (h * w)
        results.append({"quality": q, "bpp": bpp, "n_bytes": n_bytes, "decoded": decoded})
    return results
