import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fractal_compression import (
    FractalConfig, encode_channel, decode_channel, derive_leaf_block_sizes,
    encode_image, decode_image,
)
from fractal_compression.codec import EncodedChannel
from fractal_compression.metrics import psnr


def _synthetic_image(size=32, seed=0):
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 4 * np.pi, size)
    grid = np.sin(x)[:, None] * np.cos(x)[None, :] * 60 + 128
    grid += rng.normal(0, 5, size=(size, size))
    return np.clip(grid, 0, 255)


def test_grayscale_roundtrip_quality():
    img = _synthetic_image(64)
    cfg = FractalConfig(error_thresh=100)
    enc = encode_channel(img, cfg)
    recon = decode_channel(enc)
    assert recon.shape == img.shape
    assert psnr(img, recon) > 20  # lossy but should not be garbage


def test_lower_error_thresh_improves_or_matches_quality():
    img = _synthetic_image(64)
    strict = decode_channel(encode_channel(img, FractalConfig(error_thresh=20)))
    loose = decode_channel(encode_channel(img, FractalConfig(error_thresh=2000)))
    # a stricter split threshold should reconstruct at least as well
    assert psnr(img, strict) >= psnr(img, loose) - 1e-6


def test_byte_serialization_roundtrip_matches_direct_decode():
    img = _synthetic_image(48)
    cfg = FractalConfig(error_thresh=100)
    enc = encode_channel(img, cfg)
    direct = decode_channel(enc)

    raw = enc.to_bytes()
    sizes = derive_leaf_block_sizes(enc.partition_bits, enc.n_partition_bits,
                                     enc.padded_size, cfg.init_size, cfg.max_block, cfg.min_block)
    enc2 = EncodedChannel.from_bytes(raw, cfg, sizes)
    from_bytes = decode_channel(enc2)

    assert np.allclose(direct, from_bytes)


def test_color_roundtrip():
    rng = np.random.default_rng(1)
    img = rng.integers(0, 255, size=(48, 48, 3)).astype(np.float64)
    cfg = FractalConfig(error_thresh=200)
    enc = encode_image(img, cfg, chroma_subsample=True)
    recon = decode_image(enc)
    assert recon.shape == img.shape
    assert psnr(img, np.clip(recon, 0, 255)) > 10


def test_non_power_of_two_input_is_handled():
    img = _synthetic_image(50)[:50, :37]  # deliberately awkward shape
    cfg = FractalConfig(error_thresh=100)
    enc = encode_channel(img, cfg)
    recon = decode_channel(enc)
    assert recon.shape == img.shape


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
