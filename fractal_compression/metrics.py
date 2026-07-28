from __future__ import annotations
import numpy as np
from skimage.metrics import structural_similarity as ssim


def psnr(original: np.ndarray, reconstructed: np.ndarray, max_val: float = 255.0) -> float:
    mse = float(np.mean((original.astype(np.float64) - reconstructed.astype(np.float64)) ** 2))
    if mse == 0:
        return float('inf')
    return 20 * np.log10(max_val) - 10 * np.log10(mse)


def ssim_score(original: np.ndarray, reconstructed: np.ndarray) -> float:
    if original.ndim == 3:
        return float(ssim(original, reconstructed, channel_axis=2, data_range=255.0))
    return float(ssim(original, reconstructed, data_range=255.0))
