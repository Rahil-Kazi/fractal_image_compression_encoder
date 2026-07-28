import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fractal_compression import FractalConfig, encode_channel, decode_channel
from fractal_compression.metrics import psnr
from fractal_compression.codec import quantize, dequantize
from fractal_compression.encoder import _search_domain, _domain_integral_images


def _synthetic_image(size=64, seed=0):
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 4 * np.pi, size)
    grid = np.sin(x)[:, None] * np.cos(x)[None, :] * 60 + 128
    grid += rng.normal(0, 5, size=(size, size))
    return np.clip(grid, 0, 255)


def _brute_force_best_position(range_block, recon_known, config):
    rows, cols = range_block.shape
    kh, kw = recon_known.shape
    best_err, best_pos = None, None
    for y in range(kh - rows + 1):
        for x in range(kw - cols + 1):
            domain_block = recon_known[y:y + rows, x:x + cols]
            avg_domain = domain_block.mean()
            avg_range = range_block.mean()

            numerator = np.sum(domain_block * range_block) - rows * cols * avg_domain * avg_range
            denom = np.sum(domain_block ** 2) - rows * cols * avg_domain ** 2
            k_cont = numerator / denom if denom > 1e-6 else 0.0
            c_cont = avg_range - avg_domain

            k_q = dequantize(quantize(k_cont, config.k_bits, *config.k_range),
                              config.k_bits, *config.k_range)
            c_q = dequantize(quantize(c_cont, config.c_bits, *config.c_range),
                              config.c_bits, *config.c_range)

            patch = k_q * (domain_block - avg_domain) + avg_domain + c_q
            err = np.sum((range_block - patch) ** 2)
            if best_err is None or err < best_err:
                best_err, best_pos = err, (y, x)
    return best_err, best_pos


def test_quantization_aware_position_matches_brute_force():
    rng = np.random.default_rng(3)
    recon_known = rng.uniform(0, 255, size=(20, 20))
    range_block = rng.uniform(0, 255, size=(4, 4))
    config = FractalConfig(quantization_aware=True, k_bits=5, c_bits=7)

    ii, ii2 = _domain_integral_images(recon_known)
    result = _search_domain(range_block, recon_known, ii, ii2, config)

    brute_err, brute_pos = _brute_force_best_position(range_block, recon_known, config)

    assert (result.domain_row, result.domain_col) == brute_pos
    assert result.error == pytest.approx(brute_err, abs=1e-6)


def test_quantization_aware_roundtrip_quality():
    img = _synthetic_image(64)
    cfg = FractalConfig(error_thresh=100, quantization_aware=True)
    enc = encode_channel(img, cfg)
    recon = decode_channel(enc)
    assert recon.shape == img.shape
    assert psnr(img, recon) > 20


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
