"""
Color-space handling for the fractal codec.

The codec itself only understands single 2D float channels. Color images
are converted to YCbCr (ITU-R BT.601, full-range, matching what JPEG/PIL
use so comparisons are apples-to-apples) and each channel is coded
independently, optionally with the chroma planes subsampled 2x2 (4:2:0)
the way JPEG does -- this is the main knob for trading chroma fidelity
for bitrate in the color benchmarks.
"""
from __future__ import annotations
import numpy as np

# ITU-R BT.601 full-range RGB <-> YCbCr, same matrix PIL's 'YCbCr' mode uses.
_RGB_TO_YCBCR = np.array([
    [0.29900, 0.58700, 0.11400],
    [-0.168736, -0.331264, 0.50000],
    [0.50000, -0.418688, -0.081312],
])
_YCBCR_TO_RGB = np.linalg.inv(_RGB_TO_YCBCR)


def rgb_to_ycbcr(rgb: np.ndarray) -> np.ndarray:
    """rgb: (H,W,3) float in [0,255]. Returns (H,W,3) float, Y/Cb/Cr all in [0,255]."""
    rgb = rgb.astype(np.float64)
    ycbcr = rgb @ _RGB_TO_YCBCR.T
    ycbcr[..., 1:] += 128.0
    return ycbcr


def ycbcr_to_rgb(ycbcr: np.ndarray) -> np.ndarray:
    ycbcr = ycbcr.astype(np.float64).copy()
    ycbcr[..., 1:] -= 128.0
    rgb = ycbcr @ _YCBCR_TO_RGB.T
    return np.clip(rgb, 0, 255)


def subsample_chroma(channel: np.ndarray) -> np.ndarray:
    """2x2 average-pool downsample (4:2:0 style)."""
    h, w = channel.shape
    h2, w2 = h - (h % 2), w - (w % 2)
    c = channel[:h2, :w2]
    return c.reshape(h2 // 2, 2, w2 // 2, 2).mean(axis=(1, 3))


def upsample_chroma(channel: np.ndarray, out_shape: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbor 2x upsample back to out_shape (matches JPEG's typical decode)."""
    up = np.repeat(np.repeat(channel, 2, axis=0), 2, axis=1)
    H, W = out_shape
    padded = np.zeros((H, W), dtype=up.dtype)
    ph, pw = min(H, up.shape[0]), min(W, up.shape[1])
    padded[:ph, :pw] = up[:ph, :pw]
    if ph < H:
        padded[ph:, :] = padded[ph - 1, :]
    if pw < W:
        padded[:, pw:] = padded[:, pw - 1][:, None]
    return padded
